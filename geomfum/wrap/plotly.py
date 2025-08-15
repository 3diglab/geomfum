"""Wraps plotly functions."""

import plotly.graph_objects as go

from geomfum.plot import ShapePlotter
from geomfum.shape.convert import to_go_mesh3d

# TODO: add pointcloud alternatives/ enable to plot pointclouds


class PlotlyMeshPlotter(ShapePlotter):
    """Plotting object to display meshes."""

    # NB: for now assumes only one mesh is plotted

    def __init__(self, colormap="viridis"):
        self.colormap = colormap

        self._plotter = self.fig = go.Figure(
            data=[],
            layout=go.Layout(scene=dict(aspectmode="data")),
        )

    def add_mesh(self, mesh, **kwargs):
        """Add mesh to plot.

        Parameters
        ----------
        mesh : TriangleMesh
            Mesh to be plotted.
        """
        mesh3d = to_go_mesh3d(mesh)
        mesh3d.update(colorscale=self.colormap, **kwargs)

        self._plotter.update(data=[mesh3d])

        hover_text = [f"Index: {index}" for index in range(len(mesh.vertices))]
        self._plotter.data[0]["text"] = hover_text
        return self

    def set_vertex_scalars(self, scalars, name="scalars"):
        """Set vertex scalars on mesh.

        Parameters
        ----------
        scalars : array-like
            Value at each vertex.
        name : str
            Ignored.
        """
        data = self._plotter.data[0]
        data["intensity"] = scalars
        data["colorscale"] = self.colormap
        self._plotter.data[0].update(data)
        return self
    
    def highlight_vertices(self, coords, color='red', size=4,):
        """Highlight vertices on mesh.

        Parameters
        ----------
        coords : array-like, shape=[n_vertices, 3]
            Coordinates of vertices to highlight.
        color : str
            Color of the highlighted vertices as str.
        size : int
            Size of the highlighted vertices.
        """
        name = 'Highlighted_points'
        marker = go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode='markers',
            marker=dict(size=size, color=color),
            name=name
        )
        self._plotter.add_trace(marker)
        return self

    def show(self):
        """Display plot."""
        self._plotter.show()

    def save(self, filename):
        """Save plot to file.

        Parameters
        ----------
        filename : str
            Location to write file to.
        """
        self._plotter.write_image(filename)
