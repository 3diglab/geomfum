"""Definition of point cloud."""

import geomstats.backend as gs
import sklearn.neighbors as neighbors

import geomfum.backend as xgs
from geomfum.io import load_pointcloud
from geomfum.metric import HeatDistanceMetric
from geomfum.shape.shape_utils import (
    compute_edge_tangent_vectors,
    compute_gradient_matrix_fem,
    compute_tangent_frames,
)

from ._base import Shape


class PointCloud(Shape):
    """Point cloud.

    Parameters
    ----------
    vertices : array-like, shape=[n_vertices, 3]
        Vertices of the point cloud.
    """

    def __init__(self, vertices):
        super().__init__(is_mesh=False)
        self.vertices = gs.asarray(vertices)

        self.n_neighbors = 30
        self._knn_graph = None

        self._vertex_areas = None
        self._vertex_normals = None
        self._vertex_tangent_frames = None

        self._edges = None
        self._edge_tangent_vectors = None
        self._gradient_matrix = None
        self._dist_matrix = None
        self.metric = None

    @classmethod
    def from_file(cls, filename):
        """Instantiate given a file.

        Returns
        -------
        mesh : PointCloud
            A point cloud.
        """
        vertices = load_pointcloud(filename)
        return cls(vertices)

    @property
    def n_vertices(self):
        """Number of points.

        Returns
        -------
        n_vertices : int
        """
        return self.vertices.shape[0]

    # TODO: check if we can impleemnt this somewhere else ar make it a callable
    @property
    def knn_graph(self):
        """Compute k-nearest neighbors graph for the point cloud.

        Returns
        -------
        knn_info : dict
            Dictionary containing:
            - 'indices': array-like, shape=[n_vertices, k] - neighbor indices for each vertex
            - 'distances': array-like, shape=[n_vertices, k] - distances to neighbors
            - 'k': int - number of neighbors
            - 'nbrs_model': sklearn.neighbors.NearestNeighbors - fitted model for reuse
        """
        if self._knn_graph is None:
            vertices_np = gs.to_numpy(xgs.to_device(self.vertices, "cpu"))

            neigs = neighbors.NearestNeighbors(
                n_neighbors=self.n_neighbors, algorithm="kd_tree"
            ).fit(vertices_np)

            distances, indices = neigs.kneighbors(vertices_np)

            self._knn_graph = {
                "indices": indices,
                "distances": distances,
                "k": self.n_neighbors,
                "neighbors_model": neigs,
            }

        return self._knn_graph

    @property
    def vertex_normals(self):
        """Compute vertex normals for the point cloud.

        Returns
        -------
        normals : array-like, shape=[n_vertices, 3]
            Normalized per-vertex normals estimated from local neighborhoods using PCA.
        """
        if self._vertex_normals is None:
            neighbor_indices = gs.array(self.knn_graph["indices"])
            all_neighborhoods = self.vertices[neighbor_indices]
            centroids = gs.mean(all_neighborhoods, axis=1)
            local_neighborhoods = all_neighborhoods - centroids[:, None, :]
            cov_matrices = gs.einsum(
                "ijk,ijl->ikl", local_neighborhoods, local_neighborhoods
            ) / (self.n_neighbors - 1)
            try:
                _, _, v = gs.linalg.svd(cov_matrices)
                normals = v[:, :, 2]
            except Exception:
                normals = gs.zeros_like(self.vertices)
                normals[:, 2] = 1.0

            # orient normals consistently, if normal is more aligned with inward direction, flip it
            neighbor_vectors = all_neighborhoods - self.vertices[:, None, :]
            avg_neighbor_direction = gs.mean(neighbor_vectors, axis=1)
            dot_products = gs.sum(normals * avg_neighbor_direction, axis=1)
            flip_mask = dot_products > 0
            normals[flip_mask] *= -1

            # Normalize normals
            norms = gs.linalg.norm(normals, axis=1, keepdims=True)
            normals = normals / (norms + 1e-12)

            self._vertex_normals = normals

        return self._vertex_normals

    @property
    def vertex_tangent_frames(self):
        """Compute vertex tangent frame.

        Returns
        -------
        tangent_frame : array-like, shape=[n_vertices, 3, 3]
            Tangent frame of the mesh, where:
            - [n_vertices, 0, :] are the X basis vectors
            - [n_vertices, 1, :] are the Y basis vectors
            - [n_vertices, 2, :] are the vertex normals
        """
        if self._vertex_tangent_frames is None:
            self._vertex_tangent_frames = compute_tangent_frames(
                self.vertices, self.vertex_normals
            )

        return self._vertex_tangent_frames

    @property
    def edges(self):
        """Compute edges of the graph of the point cloud."""
        if self._edges is None:
            neighbor_indices = gs.array(self.knn_graph["indices"])
            edge_inds_from = gs.repeat(
                gs.arange(self.vertices.shape[0]), self.n_neighbors
            )
            self._edges = gs.stack((edge_inds_from, neighbor_indices.flatten()))

        return self._edges

    @property
    def edge_tangent_vectors(self):
        """Compute edge tangent vectors.

        Returns
        -------
        edge_tangent_vectors : array-like, shape=[n_edges, 2]
            Tangent vectors of the edges, projected onto the local tangent plane.
        """
        if self._edge_tangent_vectors is None:
            edge_tangent_vectors = compute_edge_tangent_vectors(
                self.vertices,
                self.edges,
                self.vertex_tangent_frames,
            )
            self._edge_tangent_vectors = edge_tangent_vectors
        return self._edge_tangent_vectors

    @property
    def gradient_matrix(self):
        """Compute the gradient operator as a complex sparse matrix.

        Returns
        -------
        grad_op : xgs.sparse.csc_matrix, shape=[n_vertices, n_vertices]
            Complex sparse matrix representing the gradient operator.
        """
        if self._gradient_matrix is None:
            self._gradient_matrix = compute_gradient_matrix_fem(
                self.vertices,
                self.edges,
                self.edge_tangent_vectors,
            )
        return self._gradient_matrix

    @property
    def dist_matrix(self):
        """Compute metric distance matrix.

        Returns
        -------
        _dist_matrix : array-like, shape=[n_vertices, n_vertices]
            Metric distance matrix.
        """
        if self._dist_matrix is None:
            if self.metric is None:
                raise ValueError("Metric is not set.")
            self._dist_matrix = self.metric.dist_matrix()
        return self._dist_matrix

    def equip_with_metric(self, metric):
        """Set the metric for the point cloud.

        Parameters
        ----------
        metric : class
            A metric class to use for the point cloud.
        """
        if metric == HeatDistanceMetric:
            self.metric = metric.from_registry(which="pp3d", shape=self)
        else:
            self.metric = metric(self)
        self._dist_matrix = None
