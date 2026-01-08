# defect_detection

[![PyPI](https://img.shields.io/pypi/v/defect_detection)](https://pypi.org/project/defect_detection/)
[![Build](https://github.com/lovaslin/defect_detection/actions/workflows/cd.yml/badge.svg)](https://github.com/lovaslin/defect_detection/actions)

This packge provides a basic API to implement defect detection algorithms.
Those can be tuned in order to automatically detect any defects in a PCB or other components.

## Requirement

The following packages are required :
- numpy
- opencv-python
- torch
- scikit-learn

Recommended python version >= 3.8

## Installation

To install the latest stable release from PyPI :
```bash
pip install defect_detection
```

For developper who wants to work with a local and editable version :
```bash
git clone https://github.com/lovaslin/defect_detection.git
cd defect_detection
pip install -e .
```

For the local install, you should of course run the commands using a clean python environment.
I recommend to use `venv` to setup a pip-friendly environemnt.

## Available features

- Dataset creation

The provided `defect_detection.generate_dataset` function can be used to generate a dataset suitable for training and/or testing models.
A list of the source image file names must be provided together with preprocessing and data augmetation parameters.

The specification of the function arguments are available in the doctring.

- Input batch loading

An batch of images can be loaded from disk directly using the `defect_detection.load_batch` function.
Optionnaly, it is possible produce a noisy version of the input batch than can be used e.g. for training a new model.
The batch of images is returned as a torch tensor stored on the required device.

If the data was already loaded as a numpy array, it is possible to convert it to a torch tensor using the `defect_detection.get_tensor` function.
The option to generate a noisy version is also available.

- New model training

The `defect_detection.deepAE_train` function provides a basic training loop to traina new unsupervised defect detection model.
It is recommended to use a dataset generated with the `defect_detection.generate_dataset` to perform the training (but not mandatory).
Note that a file containing the specification of the model structure hiperparameters must be provided (see specification in [wiki](https://github.com/lovaslin/defect_detection/wiki/Specification)).
The trained model will be saved on disk and the training and validation loss functions will be returned after completion of the training.

It is also possible write a custom training loop using the built-in `AE_cls.batch_train` method to compute the loss and update model parameters.

- Load a existing model for application

The `defect_detection.deepAE_load` function allows to load a previouly trained model from the disk.
By default, the model will set for application only (no training functionality available).

Once a model is loaded, it is possible to compute both the per pixel anomaly score map and loss using the built-in `AE_cls.batch_apply` method.
