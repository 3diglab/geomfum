Functional Maps
===============

Functional Maps provide a powerful framework for representing and computing correspondences between geometric shapes. This section introduces the key concepts and mathematical foundations.

Mathematical Foundation
----------------------

Functional maps represent correspondences between shapes as linear operators between function spaces. Given two shapes :math:`S_1` and :math:`S_2`, a functional map :math:`T` maps functions on :math:`S_1` to functions on :math:`S_2`.

.. math::

    T: \mathcal{F}(S_1) \rightarrow \mathcal{F}(S_2)

where :math:`\mathcal{F}(S_i)` denotes the space of real-valued functions on shape :math:`S_i`.

Laplace-Beltrami Basis
----------------------

The Laplace-Beltrami operator :math:`\Delta` is a fundamental tool in shape analysis. Its eigenfunctions :math:`\phi_i` form an orthonormal basis for the function space:

.. math::

    \Delta \phi_i = \lambda_i \phi_i

where :math:`\lambda_i` are the corresponding eigenvalues.

In GeomFuM, the Laplace-Beltrami eigenfunctions are computed using:

.. code-block:: python

    import geomfum as gfm
    
    # Compute Laplace-Beltrami eigenfunctions
    eigenfunctions, eigenvalues = gfm.laplacian.spectrum(mesh, k=100)

Functional Map Representation
----------------------------

A functional map :math:`T` can be represented as a matrix :math:`C` in the Laplace-Beltrami basis:

.. math::

    T(f) = \sum_{i,j} C_{ij} \langle f, \phi_i^{(1)} \rangle \phi_j^{(2)}

where :math:`\phi_i^{(1)}` and :math:`\phi_j^{(2)}` are the eigenfunctions of shapes :math:`S_1` and :math:`S_2` respectively.

Computing Functional Maps
------------------------

Functional maps can be computed from various types of constraints:

1. **Descriptor Preservation**: Maps that preserve shape descriptors
2. **Landmark Constraints**: Maps that align specific points
3. **Regularization**: Maps that satisfy additional geometric constraints

.. code-block:: python

    # Compute functional map from descriptors
    fm = gfm.functional_map(mesh1, mesh2, descriptors1, descriptors2)
    
    # Compute functional map with landmarks
    fm = gfm.functional_map(mesh1, mesh2, landmarks1, landmarks2)

Pointwise Correspondences
------------------------

While functional maps operate on function spaces, we often need pointwise correspondences. These can be extracted using:

.. math::

    \pi(x) = \arg\max_{y \in S_2} \sum_i \phi_i^{(2)}(y) (T\phi_i^{(1)})(x)

.. code-block:: python

    # Convert functional map to pointwise correspondences
    correspondences = gfm.pointwise_from_functional(fm, mesh1, mesh2)

Refinement and Optimization
--------------------------

Functional maps can be refined using various techniques:

- **ZoomOut**: Spectral upsampling for better correspondence quality
- **Sinkhorn Filtering**: Optimal transport-based refinement
- **Structured Regularization**: Geometric constraint enforcement

.. code-block:: python

    # Apply ZoomOut refinement
    refined_fm = gfm.refine.zoomout(fm, mesh1, mesh2)
    
    # Apply Sinkhorn filtering
    filtered_fm = gfm.refine.sinkhorn(fm, mesh1, mesh2)

Deep Learning Integration
------------------------

GeomFuM supports deep learning approaches for functional map computation:

- **Deep Functional Maps**: Learning-based descriptor extraction
- **Neural Adjoint Maps**: End-to-end correspondence learning
- **Feature Learning**: Optimizing shape representations

.. code-block:: python

    # Deep functional maps model
    model = gfm.learning.DeepFunctionalMaps()
    fm = model(mesh1, mesh2)

Applications
------------

Functional maps find applications in:

- **Shape Matching**: Finding correspondences between similar shapes
- **Shape Retrieval**: Comparing shapes in large databases
- **Shape Deformation**: Transferring deformations between shapes
- **Shape Analysis**: Understanding shape structure and properties

Key Papers Implemented
---------------------

GeomFuM implements algorithms from several key papers:

1. `Functional Maps: A Flexible Representation of Maps Between Shapes <http://www.lix.polytechnique.fr/~maks/papers/obsbg_fmaps.pdf>`_
2. `ZoomOut: Spectral Upsampling for Efficient Shape Correspondence <https://arxiv.org/abs/1904.07865>`_
3. `Deep Geometric Functional Maps: Robust Feature Learning for Shape Correspondence <https://arxiv.org/abs/2003.14286>`_
4. `Fast Sinkhorn Filters: Using Matrix Scaling for Non-Rigid Shape Correspondence <https://openaccess.thecvf.com/content/CVPR2021/html/Pai_Fast_Sinkhorn_Filters_Using_Matrix_Scaling_for_Non-Rigid_Shape_Correspondence_CVPR_2021_paper.html>`_

For more detailed examples and tutorials, see the :doc:`../tutorials/functional_maps_basics` section. 