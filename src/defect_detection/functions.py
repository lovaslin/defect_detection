#########################################
# Basic usufull functions for the API. ##
#########################################

import torch
from .deep_AE import AE_cls
from pathlib import Path
import cv2 as cv
import numpy as np


# Default loss function for model application
def loss_def(x, y):
    return torch.mean(torch.square(x - y), (1, 2, 3))


# Generic error map function
def emap(x, y, op="sum"):
    """
    Compute a error map (anomaly score per pixel) between 2 images.
    The anomaly score is computed either by averaging or summing over the 3 color channels.

    Arguments :
        x : (numpy.ndarray or torch.Tensor)
            The first input image given either as a numpy array or torch tensor.
            If it is a torch tensor, it will be automatically converted to numpy.
        y : (numpy.ndarray or torch.Tensor)
            The second input image given either as a numpy array or torch tensor.
            If it is a torch tensor, it will be automatically converted to numpy.
        op : (str)
            String specifying the operation used to compute the error map
            Can be either \"sum\" or \"mean\".
            Default to \"sum\"

    Returns :
        emap : (numpy.ndarray)
            The error map returned as a numpy array.
    """
    # Check type of first input and convert to numpy if needed
    if type(x) is torch.Tensor:
        # To numpy
        x = x.detach().to("cpu").numpy()

    # Check axis ordering of first input
    if x.shape[0] == 3:
        # Put feature axis in last position (axis=2)
        x = np.moveaxis(x, 0, 2)

    # Same for second input
    if type(y) is torch.Tensor:
        # To numpy
        y = y.detach().to("cpu").numpy()

    # Check axis ordering
    if y.shape[0] == 3:
        # Put feature axis in last position (axis=2)
        y = np.moveaxis(y, 0, 2)

    if op == "sum":
        emap = np.sum((x - y) ** 2, axis=2)
    else:
        emap = np.mean((x - y) ** 2, axis=2)

    return emap


# Deprecated function for retro-compatibility (DEPRECATED)
def emap_sum(x, y):
    """
    **DEPRECATED**
    Same as emap(x, y, \"sum\")
    """
    print(
        "WARNING : function 'emap_sum(x,y)' is deprecated, please use 'emap(x,y,\"sum\")' instead"
    )
    return emap(x, y, "sum")


def emap_mean(x, y):
    """
    **DEPRECATED**
    Same as emap(x, y, \"mean\")
    """
    print(
        "WARNING : function 'emap_mean(x,y)' is deprecated, please use 'emap(x,y,\"mean\")' instead"
    )
    return emap(x, y, "mean")


# Load a trained model
def deepAE_load(
    path, dev="auto", use_only=True, loss_fn=loss_def, opt=None, opt_param=None
):
    """
    Function to load a trained deepAE model.
    By default, the loaded model is assumed be used for application only.

    Arguments :
        path : (Path or str)
            The path to the directory where the model is stored.
            The folder must contain the hyperparameter file and the trained parameters.

        dev : (str)
            Specifiy the device where the output tensor should be strored.
            Can be either 'cuda', 'cpu' or 'auto'.
            If set to 'audo', the device is choosen automatically based on Cuda availability.
            Dafault to 'auto'

        use_only : (bool)
            Specify if the model is intended for application only.
            If true, the training method cannot be called.
            Default to True

        loss_fn : (callable)
            The loss function used to train/test the model.
            It must be compatible with the loss function requirement of pytorch.
            Ignored if use_only is set to True.
            Default to torch.mean(torch.square(x - y), (1, 2, 3))

        opt : (torch.optim or None)
            The optimizer used to trained the model.
            Ignored if use_only is set to True.
            Default to None

        opt_param : (dict or None)
            The parameters to be passed to the optimizer.
            If None, a empty dict is assumed.
            Default to None
    """
    # Load the model hyperparameter
    with Path.open(Path(path + "AE_config.txt")) as f:
        param = eval(f.read())

    # Initilize the model
    ae = AE_cls(param, use_only, loss_fn, opt, opt_param)
    ae.to_dev(dev)

    # Check device if set to auto
    if dev == "auto":
        if torch.cuda.is_available():
            dev = "cuda"
        else:
            dev = "cpu"

    # Load trained parameters
    if dev == "cuda":
        ae.load_state_dict(torch.load(path + "AE_state.save"))
    else:
        ae.load_state_dict(torch.load(path + "AE_state.save", map_location="cpu"))

    return ae


# Function to convert numpy array batch into pytorch tensor batch
def get_tensor(batch, dev="auto", npatch=0, patch_dir=None, patch_size=(10, 100)):
    """
    Conver a batch of input image from numpy array format to torch tensor on the required device.
    Optionally, you can also specify if the noising of the input is needed (e.g. for training a new model).

    Arguments :
        batch : (numpy.ndarray)
            The batch of input images given as a numpy array with shape (nbatch, size_x, size_y, nchannels).

        dev : (str)
            Specify on which device the output tensor must be stored.
            Can be either 'cpu', 'cuda' or 'auto'.
            If set to 'audo', the device is choosen automatically based on Cuda availability.
            Default to 'auto'.

        npatch : (int)
            Integer giving the number of noise patch to be added to the input.
            If greater than 0, a noisy version of the input batch is also returned.
            If 0, only the normal input batch is returned.
            Default to 0.

        patch_dir : (str or None)
            The path to the directory containing the noise pattern images to be used for noising given as a string.
            If None, random rectangular patterns will be produced.
            Default to None.

        patch_size : (iterable)
            Iterable defining the minimum and maximum size for the noise patches.
            Must be given as an iterable of 2 integers with the format [size_min, size_max].
            Default to (10, 100).

    Returns :
        batch :
            The input batch containing all images in torch tensor format.
            The output tensor shape is (nbatch, nchannels, sizex, sizey).

        batch_n : **OPTIONAL**
            The same input batch with additionnal noise patterns added to the images.
            Return only if noise patches are required.

    See also :
        load_batch
    """

    # Check for automatic device detection
    if dev == "auto":
        if torch.cuda.is_available():
            dev = "cuda"
        else:
            dev = "cpu"

    # Check if input nosing is required
    if npatch > 0:
        # Initialize the noised batch
        batch_n = batch.copy()

        # Pre-load the noise patches for path_dir (if required)
        if patch_dir is not None:
            # Initialize container
            patch = []

            # Loop over directory
            for f in Path(patch_dir).glob("*"):
                # Load noise patch
                patch.append(cv.cvtColor(cv.imread(str(f)), cv.COLOR_BGR2RGB))

        # Loop over batch images
        for i in range(batch.shape[0]):
            # Loop over npatch
            for _ in range(npatch):
                # Check if a patch directory is specified
                if patch_dir is None:
                    # Set random rectangle noise patch
                    # Size and position of patch p
                    wid = np.random.randint(patch_size[0], patch_size[1], size=2)
                    pos = [
                        np.random.randint(0, batch.shape[1] - wid[0]),
                        np.random.randint(0, batch.shape[2] - wid[1]),
                    ]

                    # Set patch color
                    if np.random.rand() < 0.5:
                        # Dark color
                        col = np.random.randint(40, 45, size=3, dtype=np.uint8)
                    else:
                        # Light color
                        col = np.random.randint(235, 240, size=3, dtype=np.uint8)

                    # Insert noise patch in image
                    batch_n[i, pos[0] : pos[0] + wid[0], pos[1] : pos[1] + wid[1]] = col
                else:
                    # Select nose partern among preloaded patches
                    # Set random size and position
                    pid = np.random.randint(0, len(patch))
                    wid = np.random.randint(patch_size[0], patch_size[1], size=2)
                    pos = [
                        np.random.randint(0, batch.shape[1] - wid[0]),
                        np.random.randint(1, batch.shape[2] - wid[1]),
                    ]

                    # Set random pi/2 rotation
                    rot = np.random.randint(-1, 3)

                    # Set random color
                    if np.random.rand() < 0.5:
                        # Dark color
                        col = np.random.randint(40, 45, size=3, dtype=np.uint8)
                    else:
                        # Light color
                        col = np.random.randint(235, 240, size=3, dtype=np.uint8)

                    # Process patch p (apply rotation and resize)
                    thisp = patch[pid].copy()
                    if rot != -1:
                        thisp = cv.rotate(thisp, rot)
                    thisp = cv.resize(thisp, (wid[1], wid[0]))

                    # Insert noise patch in image
                    batch_n[i, pos[0] : pos[0] + wid[0], pos[1] : pos[1] + wid[1], 0][
                        thisp[:, :, 0] < 250
                    ] = col[0]
                    batch_n[i, pos[0] : pos[0] + wid[0], pos[1] : pos[1] + wid[1], 1][
                        thisp[:, :, 1] < 250
                    ] = col[1]
                    batch_n[i, pos[0] : pos[0] + wid[0], pos[1] : pos[1] + wid[1], 2][
                        thisp[:, :, 2] < 250
                    ] = col[2]
                    del thisp

    # Convert batch array into a tensor with the right format
    batch = batch.astype(np.float32)
    batch = batch / 255.0
    batch = np.moveaxis(batch, 3, 1)

    # Check if need to do the same with batch_n
    if npatch > 0:
        batch_n = batch_n.astype(np.float32)
        batch_n = batch_n / 255.0
        batch_n = np.moveaxis(batch_n, 3, 1)

        # Return both batch and batch_n (in tensor format)
        return torch.from_numpy(batch).to(dev), torch.from_numpy(batch_n).to(dev)

    # Return batch in tensor format
    return torch.from_numpy(batch).to(dev)


# Load a batch of images and convert into a tensor on required device
def load_batch(flist, dev="auto", npatch=0, patch_dir=None, patch_size=(10, 100)):
    """
    Load a batch of images form a list of file path to a pytorch tensor on the required device.
    All input images are expected to have identical x and y dimensions.
    Optionally, you can also specify if the noising of the input is needed (e.g. for training a new model).

    Arguments :
        flist : (iterable)
            The list of input file pathes given as an iterable of strings.
            The format is [path_1, path_2, ..., path_n].

        dev : (str)
            Specify on which device the output tensor must be stored.
            Can be either 'cpu', 'cuda' or 'auto'.
            If set to 'audo', the device is choosen automatically based on Cuda availability.
            Default to 'auto'

        npatch : (int)
            The number of noise patch to be added to the input given as an integer.
            If greater than 0, a noisy version of the input batch is also returned.
            If 0, only the normal input batch is returned.
            Default to 0

        patch_dir : (str or None)
            The path to the directory containing the noise pattern images to be used for noising given as a string.
            If None, random rectangular patterns will be produced.
            Default to None.

        patch_size : (iterable)
            Iterable defining the minimum and maximum size for the noise patches.
            Must be given as an iterable of 2 integers with the format [size_min, size_max].
            Default to (10, 100)

    Returns :
        batch : (torch.Tensor)
            The input batch containing all images in torch tensor format.

        batch_n : (torch.Tensor) **OPTIONAL**
            The same input batch with additionnal noise patterns added to the images.
            Return only if noise patches are required.

    See also :
        get_tensor
    """

    # Load the first image and get x/y size
    img = cv.cvtColor(cv.imread(flist[0]), cv.COLOR_BGR2RGB)
    imgx, imgy = img.shape[0], img.shape[1]

    # Initialize the input array
    batch = np.empty((len(flist), imgx, imgy, 3), dtype=np.uint8)

    # Put the first image (already loaded) in container
    batch[0] = img
    del img

    # Loop over remaining files (if any) and load them
    if len(flist) > 1:
        for i, f in enumerate(flist[1:]):
            batch[i + 1] = cv.cvtColor(cv.imread(f), cv.COLOR_BGR2RGB)

    # Call get_tensor and return result
    return get_tensor(
        batch, dev=dev, npatch=npatch, patch_dir=patch_dir, patch_size=patch_size
    )


# Function to convert a tensor image batch back to numpy array (inverse get_tensor function)
def get_array(batch):
    """
    Convert a batch of images given in torch tensor format back to a numpy array.
    This is like the inverse of the get_tensor function, except that the images are returned with float type (instead of int8).

    Arguments :
        batch : (torch.Tensor)
            The batch of images given in torch.Tensor format.
            Expected tensor shape is (nbatch, nchannels, sizex, sizey)

    Returns :
        batch : (numpy.ndarray)
            The same batch of image converted into numpy array.
            The output array shape is (nbatch, sizex, sizey, nchannels).
    """
    # Convert to numpy
    batch = batch.detach().to("cpu").numpy()

    # Reorder axis (1 -> 3)
    batch = np.moveaxis(batch, 1, 3)

    return batch
