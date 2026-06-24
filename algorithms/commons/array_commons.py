"""Common array utilities used by multiple algorithms."""

import numpy as np


def populate_array(array_length):
    """Generate an array of random integers in the range [0, 99]."""
    return np.random.choice(range(100), size=array_length, replace=True)


def print_array(array):
    """Print an array followed by a blank line."""
    print(array)
    print("\n")
