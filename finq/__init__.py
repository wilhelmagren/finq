"""Initialization of base functionality of `finq`."""
from finq import datautil  # noqa
from .__version__ import __version__  # noqa
from finq.datautil.paths import _finq_working_directory_is_setup

# Check if first time importing module, then setup necessary working directories.
if not _finq_working_directory_is_setup():
    try:
        datautil.setup_finq_paths()
    except FileExistsError as e:
        print(f"Something went wrong setting up the necessary working directories: {e}")
        raise e


import numpy as np


def set_global_random_seed(seed: int):
    """Set the seed of the numpy random number generator."""
    np.random.seed(seed)