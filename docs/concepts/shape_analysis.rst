Shape Analysis
==============

Shape analysis is a fundamental component of geometric processing and functional maps. This section covers the key concepts and tools for analyzing geometric shapes.

Laplace-Beltrami Operator
-------------------------

The Laplace-Beltrami operator :math:`\Delta` is a generalization of the Laplacian to curved surfaces. It measures how much a function changes at each point of the surface:

.. math::

    \Delta f = \text{div}(\nabla f)

where :math:`\nabla` is the gradient operator and :math:`\text{div}` is the divergence operator.

Properties of the Laplace-Beltrami operator:

- **Self-adjoint**: :math:`\langle \Delta f, g \rangle = \langle f, \Delta g \rangle`
- **Positive semi-definite**: :math:`\langle \Delta f, f \rangle \geq 0`
- **Isometry invariant**: Unchanged under rigid transformations

.. code-block:: python

    import geomfum as gfm
    
    # Compute Laplace-Beltrami operator
    L = gfm.laplacian.matrix(mesh)
    
    # Compute eigenfunctions and eigenvalues
    eigenfunctions, eigenvalues = gfm.laplacian.spectrum(mesh, k=100)

Spectral Descriptors
--------------------

Spectral descriptors capture shape geometry using the spectrum of the Laplace-Beltrami operator.

Heat Kernel Signature (HKS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Heat Kernel Signature captures the amount of heat remaining at a point after time :math:`t`:

.. math::

    \text{HKS}(x, t) = \sum_{i=1}^{\infty} e^{-\lambda_i t} \phi_i^2(x)

where :math:`\lambda_i` and :math:`\phi_i` are eigenvalues and eigenfunctions.

.. code-block:: python

    # Compute HKS descriptors
    hks = gfm.descriptor.hks(mesh, t_values=[0.1, 1.0, 10.0])

Wave Kernel Signature (WKS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Wave Kernel Signature is based on the Schrödinger equation:

.. math::

    \text{WKS}(x, E) = \sum_{i=1}^{\infty} f_E^2(\lambda_i) \phi_i^2(x)

where :math:`f_E` is a Gaussian centered at energy :math:`E`.

.. code-block:: python

    # Compute WKS descriptors
    wks = gfm.descriptor.wks(mesh, energy_values=[1.0, 5.0, 10.0])

Global Point Signature (GPS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Global Point Signature combines multiple eigenvalues:

.. code-block:: python

    # Compute GPS descriptors
    gps = gfm.descriptor.gps(mesh, k=10)

Shape Descriptors
-----------------

GeomFuM provides various shape descriptors for functional map computation:

Geometric Descriptors
~~~~~~~~~~~~~~~~~~~~~

- **Gaussian Curvature**: Local surface curvature
- **Mean Curvature**: Average of principal curvatures
- **Shape Diameter Function**: Distance-based descriptor

.. code-block:: python

    # Compute geometric descriptors
    gaussian_curvature = gfm.descriptor.gaussian_curvature(mesh)
    mean_curvature = gfm.descriptor.mean_curvature(mesh)
    sdf = gfm.descriptor.shape_diameter_function(mesh)

Spectral Descriptors
~~~~~~~~~~~~~~~~~~~~

- **Laplace-Beltrami Eigenfunctions**: Basis functions for shape analysis
- **Spectral Embeddings**: Low-dimensional representations
- **Diffusion Maps**: Manifold learning approach

.. code-block:: python

    # Compute spectral descriptors
    eigenfunctions, eigenvalues = gfm.laplacian.spectrum(mesh, k=50)
    spectral_embedding = gfm.descriptor.spectral_embedding(mesh, k=10)

Feature Extraction
------------------

GeomFuM supports various feature extraction methods:

Deep Features
~~~~~~~~~~~~~

- **PointNet Features**: Deep learning features for point clouds
- **DiffusionNet Features**: Surface-based deep features
- **Custom Features**: User-defined feature extractors

.. code-block:: python

    # Extract deep features
    pointnet_features = gfm.descriptor.pointnet(mesh)
    diffusion_features = gfm.descriptor.diffusionnet(mesh)

Handcrafted Features
~~~~~~~~~~~~~~~~~~~~

- **HOG Features**: Histogram of oriented gradients
- **Spin Images**: Local surface descriptors
- **FPFH Features**: Fast Point Feature Histograms

.. code-block:: python

    # Extract handcrafted features
    hog_features = gfm.descriptor.hog(mesh)
    spin_images = gfm.descriptor.spin_images(mesh)

Shape Matching
--------------

Shape matching involves finding correspondences between similar shapes:

Isometric Matching
~~~~~~~~~~~~~~~~~~

For shapes that are approximately isometric (distance-preserving):

.. code-block:: python

    # Isometric shape matching
    correspondences = gfm.metric.isometric_matching(mesh1, mesh2)

Non-rigid Matching
~~~~~~~~~~~~~~~~~~

For shapes with non-rigid deformations:

.. code-block:: python

    # Non-rigid shape matching
    correspondences = gfm.metric.nonrigid_matching(mesh1, mesh2)

Shape Retrieval
---------------

Shape retrieval finds similar shapes in a database:

.. code-block:: python

    # Shape retrieval
    similar_shapes = gfm.metric.retrieve_similar(mesh, database, k=10)

Geometric Processing
--------------------

GeomFuM provides tools for geometric processing:

Mesh Operations
~~~~~~~~~~~~~~~

- **Mesh Simplification**: Reduce mesh complexity
- **Mesh Smoothing**: Remove noise while preserving features
- **Mesh Repair**: Fix topological issues

.. code-block:: python

    # Mesh operations
    simplified_mesh = gfm.shape.simplify(mesh, target_faces=1000)
    smoothed_mesh = gfm.shape.smooth(mesh, iterations=5)
    repaired_mesh = gfm.shape.repair(mesh)

Sampling Strategies
~~~~~~~~~~~~~~~~~~~

- **Uniform Sampling**: Random uniform sampling
- **Farthest Point Sampling**: Maximize coverage
- **Geodesic Sampling**: Based on geodesic distances

.. code-block:: python

    # Sampling strategies
    uniform_points = gfm.sample.uniform(mesh, n_points=1000)
    fps_points = gfm.sample.farthest_point(mesh, n_points=1000)
    geodesic_points = gfm.sample.geodesic(mesh, n_points=1000)

For practical examples and tutorials, see the :doc:`../tutorials/shape_analysis` section. 