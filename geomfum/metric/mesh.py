"""Module containing metrics to calculate distances on a TriangleMesh."""

import geomstats.backend as gs
import networkx as nx
from scipy.sparse.csgraph import shortest_path

import geomfum.backend as xgs
from geomfum.numerics.graph import single_source_partial_dijkstra_path_length

from ._base import FinitePointSetMetric, VertexEuclideanMetric, _SingleDispatchMixins


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
            gs.to_numpy(vertex_a), gs.to_numpy(vertex_b), gs.to_numpy(lengths)
        )
    ]

    graph = nx.Graph()
    graph.add_weighted_edges_from(weighted_edges)

    return graph


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
