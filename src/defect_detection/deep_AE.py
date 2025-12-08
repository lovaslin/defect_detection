##########################################################################
## Implementation of the super deep AE class.                           ##
## Class members :                                                      ##
##     loss_fn : the loss function used for training the model          ##
##     opt : the torch optimizer used for gradient and parameter update ##
## Class methods :                                                      ##
##     batch_train : perform one training step on a batch of inputs     ##
##     batch_apply : perform one evaluation step on a batch of inputs   ##
##########################################################################


import torch
import os


class AE_cls(torch.nn.Module):
    """
    Class inplementing the deep AE architecture using pyTorch.
    Provides also methods for training and applying model using input batches.
    """

    # Initialization method
    def __init__(self, p, apply_only=False, loss_fn=None, opt=None, opt_param=None):
        """
        Arguments :
            p : dict()
                Python dict containing all the information about the AE architecture.

            apply_only : (bool)
                Specify if the model is used only for application (default: False).
                Note that the training method is not usable if set to True.

            loss_fn : (function)
                The loss function used to train the model.
                Must be compatible with torch API.
                If you plan to use event weights, use a loss that returns one value per individual inputs.
                Not required if apply_only is set to True.

            opt : (torch.optim object)
                Instance of the torch optimizer used to compute gradient and update model parameters.
                Not required if apply_only is set to True.

            opt_param : (dict)
                Python dict containing the additionnal arguments to be passed to the optimizer.
                If not specify, no argument are given to the optimizer.
                *Optional*
        """
        super().__init__()

        dt = torch.float32

        # Function to define one encoding block
        def get_block_en(bf, ksize, drop):
            b = torch.nn.Sequential(
                torch.nn.Conv2d(
                    bf[0], bf[1], kernel_size=ksize, padding=ksize // 2, dtype=dt
                ),
                torch.nn.LeakyReLU(negative_slope=0.1),
                torch.nn.Conv2d(
                    bf[1],
                    bf[2],
                    kernel_size=ksize,
                    padding=ksize // 2,
                    stride=2,
                    dtype=dt,
                ),
                torch.nn.Dropout(drop),
                torch.nn.LeakyReLU(negative_slope=0.1),
            )
            return b

        # Function to define one decoding block
        def get_block_de(bf, ksize, drop, opad, last=False):
            if last:
                b = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(
                        bf[0],
                        bf[1],
                        kernel_size=ksize,
                        padding=ksize // 2,
                        output_padding=opad,
                        stride=2,
                        dtype=dt,
                    ),
                    torch.nn.Dropout(drop),
                    torch.nn.LeakyReLU(negative_slope=0.1),
                    torch.nn.ConvTranspose2d(
                        bf[1], bf[2], kernel_size=ksize, padding=ksize // 2, dtype=dt
                    ),
                )
            else:
                b = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(
                        bf[0],
                        bf[1],
                        kernel_size=ksize,
                        padding=ksize // 2,
                        output_padding=opad,
                        stride=2,
                        dtype=dt,
                    ),
                    torch.nn.Dropout(drop),
                    torch.nn.LeakyReLU(negative_slope=0.1),
                    torch.nn.ConvTranspose2d(
                        bf[1], bf[2], kernel_size=ksize, padding=ksize // 2, dtype=dt
                    ),
                    torch.nn.LeakyReLU(negative_slope=0.1),
                )
            return b

        # Define the encoding sequense
        en_seq = (
            get_block_en(bf, ksize, drop)
            for bf, ksize, drop in zip(p["block_size_in"], p["block_ker"], p["drop"])
        )

        # Define the decoding sequence
        de_seq = (
            get_block_de(bf, ksize, drop, opad, i + 1 == len(p["drop"]))
            for i, (bf, ksize, drop, opad) in enumerate(
                zip(
                    p["block_size_out"],
                    p["block_ker"][::-1],
                    p["drop"][::-1],
                    p["out_pad"],
                )
            )
        )

        # Set the encoder sequence
        self.encoder = torch.nn.Sequential(
            *en_seq,
            torch.nn.Conv2d(
                p["block_size_in"][-1][-1],
                p["latent_size"],
                kernel_size=p["latent_ker"],
                padding=p["latent_ker"] // 2,
                stride=2,
                dtype=dt,
            )
        )

        # Set the decoder sequence
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                p["latent_size"],
                p["block_size_out"][0][0],
                kernel_size=p["block_ker"][-1],
                padding=p["block_ker"][-1] // 2,
                stride=2,
                output_padding=p["latent_opad"],
                dtype=dt,
            ),
            *de_seq
        )

        # Store the configuration
        self.config = p

        # Set the loss function and optimiser (if needed)
        self.apply_only = apply_only
        if not apply_only:
            self.loss_fn = loss_fn
            if opt_param is None:
                opt_param = dict()
            self.opt = opt(self.parameters(), **opt_param)

        return

    # Device selection method (includes automatic selection)
    def to_dev(self, dev):
        """
        Select the device torch should use for this model.
        Supports automatic selection.

        Arguments :
            dev : (str)
                Specify the device to use for this model.
                Supported values are 'cuda', 'cpu' and 'auto'.
        """
        # Check if auto
        if dev == "auto":
            # Check if cuda is available
            if torch.cuda.is_available():
                dev = "cuda"
            else:
                dev = "cpu"

        # Set device
        self.to(dev)

        return

    # Forward pass method (internal use only)
    def forward(self, x):
        """
        Return the output of the model given input x.
        """
        return self.decoder(self.encoder(x))

    # Model training method
    def batch_train(self, x, x_n, w=None):
        """
        Train model on a batch of inputs with the possibility to use input weights.
        Arguments :
            x :
                Input batch tensor, stored on the same device than the model.
                Used as target for the loss calculation.
            x_n :
                Noisy version of the input tensor, stored on the same device than the model.
                Used as input of the model.
            w : *optional*
                The input weight tensor, stored on the same device than the model.
                Default to None (don't use input weight).
        """
        # Check if apply_only
        if self.apply_only:
            print("This model is in apply ony mode, cannot call the train method.")
            return None

        # Apply model and get output
        y = self(x_n)

        # Compute loss
        loss = self.loss_fn(x, y)

        # Check if weights must be applied
        if w is not None:
            # Ponderate mean
            loss = (w * loss).sum() / w.sum()
        else:
            # Simple mean
            loss = loss.mean()

        # Gradient and optimization step
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.detach().to("cpu").numpy()

    # Model aplication method
    def batch_apply(self, x, alt_loss_fn=None):
        """
        Apply model on a batch of inputs and return both output and loss values.
        Make sure to set the model in evaluation mode before calling this method.
        Arguments :
            x :
                Input tensor on which inference is performed (must be on same device as model).
            alt_loss_fn : *optional*
                Alternative loss function to be used for model evaluation (default to None).
                If None, the default training loss function is used.
        Returns :
            y :
                The output of the model in numpy format.
            loss :
                The corresponding loss values in numpy format.
        """
        # Skip gradient computation (not needed for inference)
        with torch.inference_mode():
            # Get output
            y = self(x)

            # Get loss (check for alternative)
            if alt_loss_fn is None:
                loss = self.loss_fn(x, y)
            else:
                loss = alt_loss_fn(x, y)

        return y.detach().to("cpu").numpy(), loss.detach().to("cpu").numpy()

    # Model save method
    def save_model(self, path):
        """
        Save both the model configuration and state in two separate files.
        Argument :
            path :
                Path to the directory in which save files are created.
        """
        # Check if path exits and try to create it if not
        if not os.path.exists(path):
            os.mkdir(path, 0o755)

        # Save config in text file
        with open(path + "AE_config.txt", "w") as f:
            print(self.config, file=f)

        # Save state_dict
        torch.save(self.state_dict(), path + "AE_state.save")

        return

    # END OF AE_cls
