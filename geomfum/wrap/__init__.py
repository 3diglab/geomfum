from geomfum._registry import (
    register_face_divergence_operator,
    register_face_orientation_operator,
    register_face_valued_gradient,
    register_heat_kernel_signature,
    register_hierarchical_mesh,
    register_laplacian_finder,
    register_wave_kernel_signature,
    register_mesh_plotter,
)
from geomfum._utils import has_package

register_laplacian_finder(
    True,
    "pyfm",
    "PyfmMeshLaplacianFinder",
    requires="pyFM",
    as_default=not has_package("robust_laplacian"),
)
register_laplacian_finder(
    True,
    "robust",
    "RobustMeshLaplacianFinder",
    requires="robust_laplacian",
    as_default=has_package("robust_laplacian"),
)
register_laplacian_finder(True, "igl", "IglMeshLaplacianFinder", requires="igl")
register_laplacian_finder(
    True, "geopext", "GeopextMeshLaplacianFinder", requires="geopext"
)

register_laplacian_finder(
    False, "robust", "RobustPointCloudLaplacianFinder", requires="robust_laplacian"
)


register_heat_kernel_signature(
    "pyfm", "PyfmHeatKernelSignature", requires="pyFM", as_default=True
)
register_wave_kernel_signature(
    "pyfm", "PyfmWaveKernelSignature", requires="pyFM", as_default=True
)

register_face_valued_gradient(
    "pyfm", "PyfmFaceValuedGradient", requires="pyFM", as_default=True
)

register_face_divergence_operator(
    "pyfm", "PyfmFaceDivergenceOperator", requires="pyFM", as_default=True
)

register_face_orientation_operator(
    "pyfm", "PyFmFaceOrientationOperator", requires="pyFM", as_default=True
)

register_hierarchical_mesh(
    "pyrmt", "PyrmtHierarchicalMesh", requires="PyRMT", as_default=True
)

register_mesh_plotter(
    "plotly", "PlotlyMeshPlotter", requires="plotly", as_default=True
)
register_mesh_plotter(
    "pyvista", "PyvistaMeshPlotter", requires="pyvista", as_default=False
)

