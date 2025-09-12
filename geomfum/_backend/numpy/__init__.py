import numpy as _np
import scipy as _scipy
from numpy import geomspace, square
import torch as _torch
from . import sparse


def scatter_sum_1d(index, src, size=None):
    shape = None if size is None else (size, 1)

    dummy_indices = _np.zeros_like(index)

    return _np.array(
        _scipy.sparse.coo_matrix(
            (src, (index, dummy_indices)),
            shape=shape,
        ).todense()
    ).flatten()


def to_device(a, device):
    return a


def argsort(a, axis=-1):
    return _np.argsort(a, axis=axis)


def to_torch(a):
    return _torch.tensor(a)


def diag(array):
    return _np.diag(array)


def pinv(array):
    """Compute the pseudo-inverse of a tensor."""
    return _np.linalg.pinv(array)
