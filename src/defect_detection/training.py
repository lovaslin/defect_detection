#########################################################################################
## Implement a training loop function to the API.                                      ##
## This function includes many fuctionnalities such as weight decay and input noising. ##
#########################################################################################

import numpy as np
import torch
import os
from .deep_AE import AE_cls
from .functions import load_batch

func_def = torch.nn.MSELoss()
param_def = {"lr": 1e-3, "betas": (0.5, 0.999)}


def deepAE_train(
    model,
    config,
    data,
    loss_fn=func_def,
    opt=torch.optim.Adam,
    opt_param=param_def,
    data_dir="",
    model_dir="",
    dev="auto",
    epochs=150,
    batch_size=None,
    ntrain=None,
    npatch=0,
    patch_dir=None,
    patch_size=(10, 100),
    shuf=False,
    lr_step=None,
    lr_gamma=0.5,
    seed=666,
):
    """
    Initialize a new unsupervised model for defect detection and train it on a given dataset.
    We recommend to use a dataset generated using the generate_dataset function.
    The train model is saved on disk and can be loaded using the deepAE_load function.

    Arguments :
        model : (str)
            Name given to the new model.
            **If a model with this name already exists, it will be overwriten.**

        config : (str)
            Path (given as a string) to the configuration file containing the model hyperparameter.

        data : (str)
            Name of the dataset to be used for training the model

        loss_fn : (callable)
            The loss function to be used for the training.
            It must be a callable compatible with torch models.
            Defalt to torch.nn.MSELoss()

        opt : (torch.optim)
            The optimizer to be used for training the model.
            Must be compatible with the torch.optim API.
            Defalut to torch.optim.Adam

        opt_param : (dict)
            Arguments to be given for initilizing the optimizer.
            Must be geven as a dict with format {'arg1': val1, ..., 'argN': valN}
            Default to {'lr': 1e-3, 'betas': (0.5, 0.999)}

        data_dir : (str)
            Path to the directory containing the training dataset given as a string.
            The directory should contain both the file with generation information and the images subdirectory.

        model_dir : (str)
            Path to the directory where the model should be saved given as a string.

        dev : (str)
            Device the should be used by torch to train the model.
            Supported values are 'cuda', 'cpu' and 'auto'.
            Default to 'auto'

        epochs : (int)
            Number of training epochs given as an integer.
            Default to 150

        batch_size : (int or None)
            Size of batch used for training and validation given as an integer.
            If None, the optimal batch size is infered from the dataset generation parameter.
            Default to None

        ntrain : (int or None)
            Number of batches to be used for the training of the model given as an integer.
            If smaller than the total number of available batches, the remaining batches wil be used for validation.
            Otherwize, validation loss will not be computed.
            Default to None

        npatch : (int)
            The number of noise patch to be added to the input given as an integer.
            If 0, noising will not be applied to the input images during training.
            Default to 0.

        patch_dir : (str or None)
            The path to the directory containing the noise pattern images to be used for noising given as a string.
            If None, random rectangular patterns will be produced.
            Default to None.

        patch_size : (iterable)
            Iterable defining the minimum and maximum size for the noise patches.
            Must be given as an iterable of 2 integers with the format [size_min, size_max].
            Default to (10, 100)

        shuf : (boul)
            Specify if the images must be shuffled before defining the training and validation batches.
            Default to False

        lr_step : (int or None)
            The number of epochs after which the learning rate is reduced given as a string.
            If None, the learning rate is kept constant during the training.
            Default to None

        lr_gamma : (float)
            Define the decay rate of the learining rate given as a float.
            The learning rate after decay is modified as follow: lr <- lr * lr_gamma
            Default to 0.5

        seed : (int)
            The seed to be used for numpy random generator (ensure reproducibility).
            Default to 666

    Returns :
        loss : (numpy.array)
            All the loss values computed on the training set given as a numpy array.
            The shape is {ntrain, batch_size}.

        loss_val : (numpy.array or None)
            All the loss values computed on the validation set given as a numpy array.
            The shape is {nval, batch_size}.
            If validation is not required, None is returned instead.
    """
    ## Initilization

    np.random.seed(seed)

    # Store function argument in dict
    param = {
        "model": model,
        "config": config,
        "data": data,
        "loss_fn": loss_fn,
        "opt": opt,
        "opt_param": opt_param,
        "data_dir": data_dir,
        "model_dir": model_dir,
        "dev": dev,
        "epochs": epochs,
        "batch_size": batch_size,
        "ntrain": ntrain,
        "npatch": npatch,
        "patch_dir": patch_dir,
        "patch_size": patch_size,
        "shuf": shuf,
        "lr_step": lr_step,
        "lr_gamma": lr_gamma,
        "seed": seed,
    }
    print("Training parameter :")
    for k, i in param.items():
        print(f"\t{k} : {i}")

    # Load dataset configuration
    print("\nLoading configuration")
    with open(param["data_dir"] + param["data"] + ".txt") as f:
        cfdat = eval(f.read())

    # Check batch size
    if param["batch_size"] is None:
        # Default to data segmentation
        batch_size = cfdat["Nseg"] ** 2

    # Load model configuration
    with open(param["config"]) as f:
        config = eval(f.read())
    print("Model configuration :")
    print(config)

    # Check device
    if param["dev"] == "auto":
        if torch.cuda.is_available():
            dev = "cuda"
            print("Cuda detected : switch to cuda device")
        else:
            dev = "cpu"
            print("Cuda not detected : switch to cpu device")

    # Check save directory
    opath = param["model_dir"] + param["model"] + "/"
    if not os.path.exists(opath):
        os.mkdir(opath)

    ## Data preparation

    print("\nPreparing data")

    # Retrive list of input files
    dpath = param["data_dir"] + param["data"] + "/"
    os.system(f"ls {dpath}* > temp.txt")
    fl = []
    with open("temp.txt") as f:
        for lin in f.readlines():
            fl.append(lin.strip())
    os.system("rm -f temp.txt")

    # Shuffle files if required
    fl = np.array(fl)
    if param["shuf"]:
        fl = np.random.shuffle(fl)

    # Gather by files by batches
    fl = fl.reshape((-1, batch_size))

    # Check ntrain
    print(f"\tfound {fl.shape[0]} batches of {fl.shape[1]} images")
    if (ntrain is not None) and (ntrain > fl.shape[0]):
        ntrain = fl.shape[0]
        print("\tWARNING : ntrain is greater than the total number of batch availabe")
        print("\tWARNING : all batches will be use for training (no validation)")
        ntrain = fl.shape[0]
    if ntrain is None:
        ntrain = fl.shape[0]

    # Initialize model
    print("\nInitializing model")
    AE = AE_cls(
        p=config,
        loss_fn=param["loss_fn"],
        opt=param["opt"],
        opt_param=param["opt_param"],
    )
    AE.to_dev(dev)

    # Initialize scheduler if needed
    if param["lr_step"] is not None:
        lrs = torch.optim.lr_scheduler.StepLR(
            AE.opt, step_size=param["lr_step"], gamma=param["lr_gamma"]
        )

    # Validation boolean
    if (ntrain is not None) and (ntrain < fl.shape[0]):
        do_val = True
    else:
        do_val = False

    # Initialize loss arrays
    loss = np.empty((param["epochs"], ntrain))
    if do_val:
        loss_val = np.empty((param["epochs"], fl.shape[0] - ntrain))

    # Epoch loop
    print("\n####TRAINING LOOP####")
    for e in range(param["epochs"]):
        print(f"epoch {e + 1} / {param['epochs']} :")

        # Training batch loop
        for b in range(ntrain):
            # Load batch
            data, data_n = load_batch(
                fl[b],
                dev=dev,
                npatch=param["npatch"],
                patch_dir=param["patch_dir"],
                patch_size=param["patch_size"],
            )

            # Train model on batch and get train loss
            loss[e, b] = AE.batch_train(data, data_n)

        # Free memory
        del data
        del data_n

        # Check if validation is required
        if do_val:
            AE.eval()

            # Validation batch loop
            for b in range(ntrain, fl.shape[0]):
                # Load batch
                data = load_batch(fl[b], dev=dev)

                # Get val loss
                _, loss_val[e, b - ntrain] = AE.batch_apply(data)
            AE.train()

            # Free memory again
            del data

        # Update scheduler (if needed)
        if param["lr_step"] is not None:
            lrs.step()

        # Print epoch mean loss
        if do_val:
            print(f"\tloss = {loss[e].mean():.3g}\tval = {loss_val[e].mean():.3g}")
        else:
            print(f"\tloss = {loss[e].mean():.3g}")

    # Check save dir and save model
    print(f"\nSaving model under '{opath}save/'")
    if not os.path.exists(opath + "save/"):
        os.mkdir(opath + "save/", 0o755)
    AE.save_model(opath + "save/")

    # Return loss arrays
    if do_val:
        return (loss, loss_val)
    return (loss, None)
