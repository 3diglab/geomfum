"""Wraps polyscope functions."""

import polyscope as ps

import geomstats.backend as gs
from geomfum.plot import ShapePlotter


class PsMeshPlotter(ShapePlotter):
    """Plotting object to display meshes."""

    # NB: for now assumes only one mesh is plotted

    def __init__(self, colormap="viridis", backend=""):
        super().__init__()

        self.colormap = colormap

        self._plotter = ps
        self._name = "Mymesh"

        self._plotter.init(backend)

    def add_mesh(self, mesh):
        """Add mesh to plot.

        Parameters
        ----------
        mesh : TriangleMesh
            Mesh to be plotted.
        """
        self._plotter.register_surface_mesh(self._name, gs.to_numpy(mesh.vertices), gs.to_numpy( mesh.faces))
        return self

    def set_vertex_scalars(self, scalars, name="scalars"):
        """Set vertex scalars on mesh.

        Parameters
        ----------
        scalars : array-like
            Value at each vertex.
        name : str
            Scalar field name.
        """
        ps.get_surface_mesh(self._name).add_scalar_quantity(
            name,
            scalars,
            defined_on="vertices",
            cmap=self.colormap,
            enabled=True,
        )
        return self

    def highlight_vertices(self, coords, color=(1.0, 0.0, 0.0), size=0.01,):
        """
        Highlight vertices on a mesh using Polyscope by adding a point cloud.

        Parameters
        ----------
        coords : array-like, shape = [n_vertices, 3]
            Coordinates of vertices to highlight.
        color : tuple
            Color of the highlighted vertices (e.g., (1.0, 0.0, 0.0)).
        radius : float
            Radius of the rendered points (visual size).
        """
        name = 'Highlighted_points'
        self._plotter.register_point_cloud(name, coords, radius = size, color = color)
        return self

    def show(self):
        """Display plot."""
        self._plotter.show()

    def save(self, filename):
        """Save plot to file.

        Parameters
        ----------
        filename : str | Path | io.BytesIO
            Location to write file to.
        """
        self._plotter.screenshot(filename)
