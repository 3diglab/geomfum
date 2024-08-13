from geomfun._registry import LaplacianFinderRegistry
from geomfun.basis import LaplaceEigenBasis
from geomfun.numerics.eig import ScipyEigsh


class LaplacianFinder:
    """Algorithm to find the Laplacian.

    Parameters
    ----------
    mesh : bool
        If mesh or point cloud.
    which : str
        Which algorithm/library to use.
    """

    def __new__(cls, mesh=True, which="robust", **kwargs):
        return LaplacianFinderRegistry.MAP[(mesh, which)](**kwargs)


class LaplacianSpectrumFinder:
    """Algorithm to find Laplacian spectrum.

    Parameters
    ----------
    spectrum_size : int
        Spectrum size. Ignored if ``eig_solver`` is not None.
    nonzero : bool
        Remove zero zero eigenvalue.
    fix_sign : bool
        Wheather to have all the first components with positive sign.
    laplacian_finder : BaseLaplacianFinder
        Algorithm to find the Laplacian. Ignored if Laplace and mass matrices
        were already computed.
    eig_solver : EigSolver
        Eigen solver.
    """

    def __init__(
        self,
        spectrum_size=100,
        nonzero=False,
        fix_sign=False,
        laplacian_finder=None,
        eig_solver=None,
    ):
        if eig_solver is None:
            eig_solver = ScipyEigsh(spectrum_size=spectrum_size, sigma=-0.01)

        self.nonzero = nonzero
        self.fix_sign = fix_sign
        self.laplacian_finder = laplacian_finder
        self.eig_solver = eig_solver

    @property
    def spectrum_size(self):
        """Spectrum size.

        Returns
        -------
        spectrum_size : int
            Spectrum size.
        """
        return self.eig_solver.spectrum_size

    @spectrum_size.setter
    def spectrum_size(self, spectrum_size):
        """Set spectrum size.

        Parameters
        ----------
        spectrum_size : int
            Spectrum size.
        """
        self.eig_solver.spectrum_size = spectrum_size

    def __call__(self, shape, as_basis=True, recompute=False):
        """Apply algorithm.

        Parameters
        ----------
        shape : Shape
            Shape.
        as_basis : bool
            Whether return basis or eigenvals/vecs.
        recompute : bool
            Whether to recompute Laplacian if information is cached.

        Returns
        -------
        eigenvals : array-like, shape=[spectrum_size]
            Eigenvalues. (If ``basis is False``.)
        eigenvecs : array-like, shape=[n_vertices, spectrum_size]
            Eigenvectors. (If ``basis is False``.)
        basis : LaplaceEigenBasis
            A basis. (If ``basis is True``.)
        """
        laplace_matrix, mass_matrix = shape.find_laplacian(
            self.laplacian_finder, recompute=recompute
        )

        eigenvals, eigenvecs = self.eig_solver(laplace_matrix, M=mass_matrix)

        if self.nonzero:
            eigenvals = eigenvals[1:]
            eigenvecs = eigenvecs[:, 1:]

        if self.fix_sign:
            indices = eigenvecs[0, :] < 0
            eigenvals[indices] *= -1
            eigenvecs[:, indices] *= -1

        if as_basis:
            return LaplaceEigenBasis(shape, eigenvals, eigenvecs)

        return eigenvals, eigenvecs
