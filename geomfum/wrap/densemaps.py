from densemaps.torch import maps
import torch
import torch.nn as nn

from geomfum.convert import BaseNeighborFinder


class DenseMapsSoftmaxNeighborFinder(BaseNeighborFinder, nn.Module):
    """Softmax neighbor finder using DenseMaps.

    Finds neighbors using softmax regularization with DenseMaps.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors.
    tau : float
        Temperature parameter for softmax regularization.
    """

    def __init__(self, n_neighbors=1, tau=0.07):
        BaseNeighborFinder.__init__(self, n_neighbors=n_neighbors)
        nn.Module.__init__(self)
        self.tau = tau

    def fit(self, X, y=None):
        """Store the reference points."""
        self.X = X
        return self

    def kneighbors(self, Y, return_distance=True):
        """Find k nearest neighbors using DenseMaps softmax regularization.

        Parameters
        ----------
        Y : array-like, shape=[n_points_y, n_features]
            Query points.
        return_distance : bool
            Whether to return the distances.
        """
        P12 = self.forward(self.X, Y)
        if return_distance:
            return torch.linalg.norm(Y - P12 @ self.X, -1), P12.get_nn()[:, None]
        return P12.get_nn()[:, None]

    def forward(self, X, Y):
        """Compute the permutation matrix P as a softmax of the similarity.

        Parameters
        ----------
        X : array-like, shape=[n_points_x, n_features]
            Reference points.
        Y : array-like, shape=[n_points_y, n_features]
            Query points.

        Returns
        -------
        P : array-like, shape=[n_points_y, n_points_x]
            Permutation matrix, where each row sums to 1.
        """
        return maps.KernelDistMap(
            X, Y, blur=self.tau
        )  # A "dense" kernel map, not used in memory
