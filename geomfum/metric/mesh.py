"""Module containing metrics to calcualte distances on a mesh."""

import abc

import geomstats.backend as gs
import networkx as nx
from scipy.sparse.csgraph import shortest_path
import geomfum.backend as xgs
import networkx as nx

from geomfum._registry import HeatDistanceMetricRegistry, WhichRegistryMixins
from geomfum.numerics.graph import single_source_partial_dijkstra_path_length


def to_nx_edge_graph(shape):
    """Convert a shape to a networkx graph.

    Parameters
    ----------
    shape : Shape
        Shape.

    Returns
    -------
    graph : networkx.Graph
        Graph.
    """
    # TODO: move to utils? circular imports
    vertex_a, vertex_b = shape.edges.T
    lengths = VertexEuclideanMetric(shape).dist(vertex_a, vertex_b)

    weighted_edges = [
        (vertex_a_, vertex_b_, length)
        for vertex_a_, vertex_b_, length in zip(
            gs.to_numpy(xgs.to_device(vertex_a, "cpu")),
            gs.to_numpy(xgs.to_device(vertex_b, "cpu")),
            gs.to_numpy(xgs.to_device(lengths, "cpu")),
        )
    ]

    graph = nx.Graph()
    graph.add_weighted_edges_from(weighted_edges)

    return graph


class Metric(abc.ABC):
    """Metric.

    Parameters
    ----------
    shape : Shape
        Considered as a manifold.
    """

    def __init__(self, shape):
        self._shape = shape

    @abc.abstractmethod
    def dist(self, point_a, point_b):
        """Distance between points.

        Parameters
        ----------
        point_a : array-like, shape=[...]
            Index Point.
        point_b : array-like, shape=[...]
            Index Point.

        Returns
        -------
        dist : array-like, shape=[...,]
            Distance.
        """


class FinitePointSetMetric(Metric, abc.ABC):
    """Metric on a finite set of indexed points."""

    @abc.abstractmethod
    def dist_matrix(self):
        """Distance between all the points of a shape.

        Returns
        -------
        dist_matrix : array-like, shape=[n_vertices, n_vertices]
            Distance matrix.
        """

    @abc.abstractmethod
    def dist_from_source(self, source_point):
        """Distance from source point.

        Parameters
        ----------
        source_point : array-like, shape=[...]
            Index of source point.

        Returns
        -------
        dist : array-like, shape=[...] or list-like[array-like]
            Distance.
        target_point : array-like, shape=[n_targets] or list-like[array-like]
            Target index.
        """


class VertexEuclideanMetric(FinitePointSetMetric):
    """Euclidean metric between vertices of a mesh."""

    def dist(self, point_a, point_b):
        """Distance between mesh vertices.

        Parameters
        ----------
        point_a : array-like, shape=[...]
            Index of source point.
        point_b : array-like, shape=[...]
            Index of target point.

        Returns
        -------
        dist : array-like, shape=[...]
            Distance.
        """
        vertices = self._shape.vertices

        diff = vertices[point_a] - vertices[point_b]
        return gs.linalg.norm(diff, axis=diff.ndim - 1)

    def dist_from_source(self, source_point):
        """Distance from source point.

        Parameters
        ----------
        source_point : array-like, shape=[...]
            Index of source point.

        Returns
        -------
        dist : array-like, shape=[...] or array-like[array-like]
            Distance.
        target_point : array-like, shape=[n_targets] or array-like[array-like]
            Target index.
        """
        vertices = self._shape.vertices

        source_vertices = vertices[source_point]
        if source_vertices.ndim > 1:
            source_vertices = gs.expand_dims(source_vertices, 1)

        diff = source_vertices - vertices

        dist = gs.linalg.norm(diff, axis=diff.ndim - 1)

        target_point = gs.arange(self._shape.n_vertices)
        if diff.ndim > 1:
            target_point = gs.broadcast_to(
                target_point, dist.shape[:-1] + target_point.shape
            )

        return dist, target_point

    def dist_matrix(self):
        """Distance between mesh vertices.

        Returns
        -------
        dist_matrix : array-like, shape=[n_vertices, n_vertices]
            Distance matrix.
        """
        return self.dist_from_source(gs.arange(self._shape.n_vertices))[0]


class _SingleDispatchMixins:
    def dist(self, point_a, point_b):
        """Distance between mesh vertices.

        Parameters
        ----------
        point_a : array-like, shape=[...]
            Index of source point.
        point_b : array-like, shape=[...]
            Index of target point.

        Returns
        -------
        dist : array-like, shape=[...,]
            Distance.
        """
        point_a = gs.asarray(point_a)
        point_b = gs.asarray(point_b)

        if point_a.ndim == 0 and point_b.ndim == 0:
            return self._dist_single(point_a, point_b)

        point_a, point_b = gs.broadcast_arrays(point_a, point_b)
        return gs.stack(
            [
                self._dist_single(point_a_, point_b_)
                for point_a_, point_b_ in zip(point_a, point_b)
            ]
        )

    def dist_from_source(self, source_point):
        """Distance between mesh vertices.

        Parameters
        ----------
        source_point : array-like, shape=[...]
            Index of source point.

        Returns
        -------
        dist : array-like, shape=[...,] or list[array-like]
            Distance.
        target_point : array-like, shape=[n_targets,] or list[array-like]
            Target index.
        """
        source_point = gs.asarray(source_point)
        if source_point.ndim == 0:
            return self._dist_from_source_single(source_point)

        out = [
            self._dist_from_source_single(source_index_)
            for source_index_ in source_point
        ]
        return list(zip(*out))

    @abc.abstractmethod
    def _dist_from_source_single(self, source_point):
        pass

    @abc.abstractmethod
    def _dist_single(self, point_a, point_b):
        pass


class _NxDijkstraMixins(_SingleDispatchMixins):
    def dist_matrix(self):
        """Distance between mesh vertices.

        Returns
        -------
        dist_matrix : array-like, shape=[n_vertices, n_vertices]
            Distance matrix.

        Notes
        -----
        * infinitely slow
        """
        all_pairs = nx.all_pairs_dijkstra_path_length(self._graph)

        n_vertices = self._shape.n_vertices
        dist_mat = gs.empty((n_vertices, n_vertices))

        for node_index, all_dict in all_pairs:
            dists = gs.array(list(all_dict.values()))
            indices = gs.array(list(all_dict.keys()))
            dist_mat[node_index, indices] = dists

        return dist_mat

    def _dist_single(self, point_a, point_b):
        """Distance between mesh vertices.

        Parameters
        ----------
        point_a : array-like, shape=()
            Index of source point.
        point_b : array-like, shape=()
            Index of target point.

        Returns
        -------
        dist : numeric
            Distance.
        """
        try:
            dist, _ = nx.single_source_dijkstra(
                self._graph,
                point_a.item(),
                target=point_b.item(),
                cutoff=None,
                weight="weight",
            )
        except nx.NetworkXNoPath:
            dist = float("inf")
        return gs.asarray(dist)


class GraphShortestPathMetric(_NxDijkstraMixins, FinitePointSetMetric):
    """Shortest path on edge graph of mesh with single source Dijkstra.

    Parameters
    ----------
    shape : Shape
        Shape.
    cutoff : float
        Length (sum of edge weights) at which the search is stopped.
    """

    # TODO: add scipy-based implementation?

    def __init__(self, shape, cutoff=None):
        self.cutoff = cutoff

        super().__init__(shape)
        self._graph = to_nx_edge_graph(shape)

    def _dist_from_source_single(self, source_point):
        """Distance between mesh vertices.

        Parameters
        ----------
        source_point : array-like, shape=()
            Index of source point.

        Returns
        -------
        dist : array-like, shape=[n_targets]
            Distance.
        target_point : array-like, shape=[n_targets]
            Target index.

        Notes
        -----
        The Distances are ordered following the order of the indices.
        """
        dist_dict = nx.single_source_dijkstra_path_length(
            self._graph, source_point.item(), cutoff=self.cutoff, weight="weight"
        )
        indices = gs.asarray(list(dist_dict.keys()))
        distances = gs.asarray(list(dist_dict.values()))
        sort_order = xgs.argsort(indices)
        return gs.asarray(list(distances[sort_order])), gs.asarray(
            list(indices[sort_order])
        )


class KClosestGraphShortestPathMetric(_NxDijkstraMixins, FinitePointSetMetric):
    """Shortest path on edge graph of mesh with Dijkstra.

    Parameters
    ----------
    shape : Shape
        Shape.
    k_closest : int
        Number of nodes to find distances to (including the source itself).
    """

    def __init__(self, shape, k_closest=5):
        self.k_closest = k_closest

        super().__init__(shape)
        self._graph = to_nx_edge_graph(shape)

    def _dist_from_source_single(self, source_point):
        """Distance between mesh vertices.

        Parameters
        ----------
        source_point : array-like, shape=()
            Index of source point.

        Returns
        -------
        dist : array-like, shape=[n_closest]
            Distance.
        target_point : array-like, shape=[n_closest,]
            Target index.
        """
        dist_dict = single_source_partial_dijkstra_path_length(
            self._graph, source_point.item(), self.k_closest, weight="weight"
        )
        return gs.array(list(dist_dict.values())), gs.array(list(dist_dict.keys()))


class HeatDistanceMetric(WhichRegistryMixins):
    """Heat distance metric between vertices of a mesh.

    References
    ----------
    .. [CWW2017] Crane, K., Weischedel, C., Wardetzky, M., 2017.
        The heat method for distance computation. Commun. ACM 60, 90–99.
        https://doi.org/10.1145/3131280
    """

    _Registry = HeatDistanceMetricRegistry


class _ScipyShortestPathMixins(_SingleDispatchMixins):
    def dist_matrix(self):
        """Distance between mesh vertices.

        Returns
        -------
        dist_matrix : array-like, shape=[n_vertices, n_vertices]
            Distance matrix.

        Notes
        -----
        * infinitely slow
        """
        dist_mat = shortest_path(
            nx.adjacency_matrix(
                self._graph, nodelist=range(self._shape.vertices.shape[0])
            ).tolil(),
            directed=False,
        )

        return gs.array(dist_mat)


class ScipyGraphShortestPathMetric(_ScipyShortestPathMixins, FinitePointSetMetric):
    """Shortest path on edge graph of mesh with Scipy shortest path solver.

    Parameters
    ----------
    shape : Shape
        Shape.
    cutoff : float
        Length (sum of edge weights) at which the search is stopped.
    """

    def __init__(self, shape, cutoff=None):
        self.cutoff = cutoff

        super().__init__(shape)
        self._graph = to_nx_edge_graph(shape)

    def _dist_from_source_single(self, source_point):
        """Distance between mesh vertices.

        Parameters
        ----------
        source_point : array-like, shape=()
            Index of source point.

        Returns
        -------
        dist : array-like, shape=[n_targets]
            Distance.
        target_point : array-like, shape=[n_targets]
            Target index.
        """
        dist = shortest_path(
            nx.adjacency_matrix(
                self._graph, nodelist=range(self._shape.vertices.shape[0])
            ).tolil(),
            directed=False,
            indices=source_point,
        )
        return gs.array(list(dist)), gs.arange(len(dist))
