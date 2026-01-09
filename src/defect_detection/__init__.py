#!/usr/bin/env python

from .deep_AE import AE_cls
from .filtering import get_pixels
from .functions import (
    emap,
    emap_mean,
    emap_sum,
    deepAE_load,
    load_batch,
    get_tensor,
    get_array,
)
from .preprocessing import generate_dataset
from .training import deepAE_train

# Automatic versioning
from .version import version as __version__

__all__ = [
    "AE_cls",
    "get_pixels",
    "emap",
    "emap_mean",
    "emap_sum",
    "deepAE_load",
    "get_tensor",
    "get_array",
    "load_batch",
    "generate_dataset",
    "deepAE_train",
    "__version__",
]
