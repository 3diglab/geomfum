"""Definition of point cloud."""

import geomstats.backend as gs
import geomfum.backend as xgs

from geomfum.io import load_pointcloud

from ._base import Shape

import scipy.spatial
import sklearn.neighbors as neighbors


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
        self._vertex_normals = None
        self._vertex_tangent_frames = None
        self._gradient_matrix = None
        self._dist_matrix = None
        self.metric = None
        self.n_neighbors = 30
        self._knn_graph = None

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

            nbrs = neighbors.NearestNeighbors(
                n_neighbors=self.n_neighbors, algorithm="kd_tree"
            ).fit(vertices_np)

            distances, indices = nbrs.kneighbors(vertices_np)

            self._knn_graph = {
                "indices": indices,
                "distances": distances,
                "k": self.n_neighbors,
                "nbrs_model": nbrs,
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
            normals = self.vertex_normals
            device = getattr(normals, "device", None)

            tangent_frame = xgs.to_device(
                gs.zeros((self.n_vertices, 3, 3)), device=device
            )

            tangent_frame[:, 2, :] = normals

            basis_cand1 = xgs.to_device(
                gs.tile([1, 0, 0], (self.n_vertices, 1)), device=device
            )
            basis_cand2 = xgs.to_device(
                gs.tile([0, 1, 0], (self.n_vertices, 1)), device=device
            )

            dot_products = gs.sum(normals * basis_cand1, axis=1, keepdims=True)
            basis_x = gs.where(gs.abs(dot_products) < 0.9, basis_cand1, basis_cand2)

            normal_projections = (
                gs.sum(basis_x * normals, axis=1, keepdims=True) * normals
            )
            basis_x = basis_x - normal_projections

            basis_x_norm = gs.linalg.norm(basis_x, axis=1, keepdims=True)
            basis_x = basis_x / (basis_x_norm + 1e-12)

            basis_y = gs.cross(normals, basis_x)

            tangent_frame[:, 0, :] = basis_x
            tangent_frame[:, 1, :] = basis_y

            self._vertex_tangent_frames = tangent_frame

        return self._vertex_tangent_frames

    @property
    def gradient_matrix(self):
        """Compute the gradient operator as a complex sparse matrix.

        Returns
        -------
        grad_op : xgs.sparse.csc_matrix, shape=[n_vertices, n_vertices]
            Complex sparse matrix representing the gradient operator.
        """
        """Compute the gradient operator as a complex sparse matrix.

        Returns
        -------
        grad_op : xgs.sparse.csc_matrix, shape=[n_vertices, n_vertices]
            Complex sparse matrix representing the gradient operator.
        """
        if self._gradient_matrix is None:
            neighbor_indices = gs.array(self.knn_graph["indices"])
            # build edges
            edge_inds_from = gs.repeat(
                gs.arange(self.vertices.shape[0]), self.n_neighbors
            )
            edges = gs.stack((edge_inds_from, neighbor_indices.flatten()))

            edge_vecs = self.vertices[edges[:, 1], :] - self.vertices[edges[:, 0], :]

            basis_x = self.vertex_tangent_frames[edges[:, 0], 0, :]
            basis_y = self.vertex_tangent_frames[edges[:, 0], 1, :]

            # Project edge vectors onto the local tangent plane
            comp_x = gs.sum(edge_vecs * basis_x, axis=1)
            comp_y = gs.sum(edge_vecs * basis_y, axis=1)

            edge_tangent_vectors = gs.stack((comp_x, comp_y), axis=-1)

            outgoing_edges_per_vertex = [[] for _ in range(self.n_vertices)]
            for edge_index in range(edges.shape[0]):
                tail_ind = edges[edge_index, 0]
                tip_ind = edges[edge_index, 1]
                if tip_ind != tail_ind:
                    outgoing_edges_per_vertex[tail_ind].append(edge_index)

            row_inds = []
            col_inds = []
            data_vals = []
            eps_reg = 1e-5

            # For each vertex, fit a local linear function 'f' to its neighbors
            for vertex_idx in range(self.n_vertices):
                num_neighbors = len(outgoing_edges_per_vertex[vertex_idx])

                if num_neighbors == 0:
                    continue

                # Set up the least squares system for the local neighborhood
                lhs_mat = gs.zeros((num_neighbors, 2))  # Edge tangent vectors
                rhs_mat = gs.zeros(
                    (num_neighbors, num_neighbors + 1)
                )  # Finite Difference matrix rhs_mat[i,j] = f(j) - f(i)
                lookup_vertices_idx = [vertex_idx]

                # for each row of the rhs_mat, we have the following:
                # - rhs_mat[i, 0] = -f(center) (the value at the center vertex)
                # - rhs_mat[i, i + 1] = +f(neighbor) (the value at the neighbor vertex)
                # - rhs_mat[i, j] = 0 for j != 0, i + 1 (no other values)
                for neighbor_index in range(num_neighbors):
                    edge_index = outgoing_edges_per_vertex[vertex_idx][neighbor_index]
                    neigbor_vertex_idx = edges[edge_index, 1]
                    lookup_vertices_idx.append(neigbor_vertex_idx)

                    edge_vec = edge_tangent_vectors[edge_index][:]

                    lhs_mat[neighbor_index][:] = edge_vec
                    rhs_mat[neighbor_index][0] = -1
                    rhs_mat[neighbor_index][neighbor_index + 1] = 1

                # Solve
                lhs_T = lhs_mat.T
                lhs_inv = gs.linalg.inv(lhs_T @ lhs_mat + eps_reg * gs.eye(2)) @ lhs_T
                sol_mat = lhs_inv @ rhs_mat
                sol_coefs = gs.transpose((sol_mat[0, :] + 1j * sol_mat[1, :]))

                for i_neigh in range(num_neighbors + 1):
                    i_glob = lookup_vertices_idx[i_neigh]
                    row_inds.append(vertex_idx)
                    col_inds.append(i_glob)
                    data_vals.append(sol_coefs[i_neigh])

            # Build the sparse matrix
            row_inds = gs.asarray(row_inds)
            col_inds = gs.asarray(col_inds)
            data_vals = gs.asarray(data_vals)

            self._gradient_matrix = xgs.sparse.to_csc(
                xgs.sparse.coo_matrix(
                    gs.stack([row_inds, col_inds]),
                    data_vals,
                    shape=(self.n_vertices, self.n_vertices),
                )
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
        # TODO: Implement metric assignment for point clouds
        self.metric = metric(self)
        self._dist_matrix = None
