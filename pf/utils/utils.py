import numpy as np
from itertools import chain, zip_longest


def alt_chain(*iters, fillvalue=None):
    """returns a list of alternating values between the input lists/arrays"""
    return chain.from_iterable(zip_longest(*iters, fillvalue=fillvalue))


def split_given_size(a, size):
    """Split an array given the size and then appending the remainder"""
    return np.split(a, np.arange(size, len(a), size))
