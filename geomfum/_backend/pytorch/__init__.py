import geomstats.backend as gs
import torch as _torch
from torch import square

from . import sparse


def scatter_sum_1d(index, src, size=None):
    if size is None:
        size = index.max() + 1

    array = _torch.zeros(size, dtype=src.dtype)
    return _torch.scatter_add(array, -1, index, src)


def geomspace(start, stop, num, *, dtype=None):
    return gs.exp(gs.linspace(gs.log(start), gs.log(stop), num))


def argsort(a, axis=-1):
    return _torch.argsort(a, dim=axis)


def to_torch(a):
    return a


def to_device(a, device):
    return a.to(device)


def diag(array):
    if array.ndim == 1:
        return _torch.diag(array)
    elif array.ndim == 2:
        return _torch.diag(_torch.diagonal(array))
    else:
        raise ValueError("Input must be a 1D or 2D tensor.")


def pinv(array):
    """Compute the pseudo-inverse of a tensor."""
    return _torch.linalg.pinv(array)
