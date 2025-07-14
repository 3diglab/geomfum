Shape Correspondence
====================

Shape correspondence is the problem of finding meaningful point-to-point mappings between geometric shapes. This section covers the fundamental concepts and evaluation methods.

Types of Correspondences
-----------------------

Pointwise Correspondences
~~~~~~~~~~~~~~~~~~~~~~~~~

Pointwise correspondences map each point on one shape to a point on another shape:

.. math::

    \pi: S_1 \rightarrow S_2

where :math:`\pi(x)` is the corresponding point of :math:`x` on shape :math:`S_2`.

.. code-block:: python

    import geomfum as gfm
    
    # Compute pointwise correspondences
    correspondences = gfm.pointwise_from_functional(functional_map, mesh1, mesh2)

Functional Correspondences
~~~~~~~~~~~~~~~~~~~~~~~~~~

Functional correspondences map functions between shapes:

.. math::

    T: \mathcal{F}(S_1) \rightarrow \mathcal{F}(S_2)

.. code-block:: python

    # Compute functional correspondences
    functional_map = gfm.functional_map(mesh1, mesh2, descriptors1, descriptors2)

Dense Correspondences
~~~~~~~~~~~~~~~~~~~~

Dense correspondences provide mappings for every point on the surface:

.. code-block:: python

    # Compute dense correspondences
    dense_correspondences = gfm.dense_correspondence(mesh1, mesh2)

Correspondence Quality
---------------------

Geometric Distortion
~~~~~~~~~~~~~~~~~~~~

Geometric distortion measures how well the correspondence preserves geometric properties:

.. math::

    \text{Distortion} = \frac{d_{S_1}(x, y)}{d_{S_2}(\pi(x), \pi(y))}

where :math:`d_{S_i}` is the geodesic distance on shape :math:`S_i`.

.. code-block:: python

    # Compute geometric distortion
    distortion = gfm.metric.geometric_distortion(correspondences, mesh1, mesh2)

Isometric Error
~~~~~~~~~~~~~~~

Isometric error measures deviation from isometry:

.. code-block:: python

    # Compute isometric error
    isometric_error = gfm.metric.isometric_error(correspondences, mesh1, mesh2)

Conformal Error
~~~~~~~~~~~~~~~

Conformal error measures angle preservation:

.. code-block:: python

    # Compute conformal error
    conformal_error = gfm.metric.conformal_error(correspondences, mesh1, mesh2)

Correspondence Evaluation
------------------------

Ground Truth Comparison
~~~~~~~~~~~~~~~~~~~~~~

When ground truth correspondences are available:

.. code-block:: python

    # Compare with ground truth
    accuracy = gfm.metric.correspondence_accuracy(predicted, ground_truth)
    geodesic_error = gfm.metric.geodesic_error(predicted, ground_truth, mesh2)

Landmark Evaluation
~~~~~~~~~~~~~~~~~~~

Using landmark correspondences for evaluation:

.. code-block:: python

    # Evaluate using landmarks
    landmark_error = gfm.metric.landmark_error(correspondences, landmarks1, landmarks2)

Correspondence Refinement
------------------------

ZoomOut Algorithm
~~~~~~~~~~~~~~~~~

ZoomOut refines correspondences by spectral upsampling:

.. code-block:: python

    # Apply ZoomOut refinement
    refined_correspondences = gfm.refine.zoomout(correspondences, mesh1, mesh2)

Sinkhorn Filtering
~~~~~~~~~~~~~~~~~~

Sinkhorn filtering uses optimal transport for refinement:

.. code-block:: python

    # Apply Sinkhorn filtering
    filtered_correspondences = gfm.refine.sinkhorn(correspondences, mesh1, mesh2)

Iterative Refinement
~~~~~~~~~~~~~~~~~~~

Iterative refinement combines multiple techniques:

.. code-block:: python

    # Iterative refinement
    refined_correspondences = gfm.refine.iterative(
        correspondences, mesh1, mesh2, iterations=10
    )

Correspondence Applications
--------------------------

Shape Transfer
~~~~~~~~~~~~~~

Transfer properties between shapes:

.. code-block:: python

    # Transfer texture or deformation
    transferred_property = gfm.transfer.property(
        source_mesh, target_mesh, correspondences, property_values
    )

Shape Interpolation
~~~~~~~~~~~~~~~~~~~

Interpolate between shapes:

.. code-block:: python

    # Shape interpolation
    interpolated_mesh = gfm.interpolate.shapes(
        mesh1, mesh2, correspondences, t=0.5
    )

Shape Registration
~~~~~~~~~~~~~~~~~~

Register shapes to a common reference:

.. code-block:: python

    # Shape registration
    registered_mesh = gfm.registration.align(mesh, reference_mesh, correspondences)

Correspondence Visualization
---------------------------

GeomFuM provides various visualization tools:

Correspondence Lines
~~~~~~~~~~~~~~~~~~~~

Visualize correspondences as lines between shapes:

.. code-block:: python

    # Visualize correspondences
    gfm.plot.correspondences(mesh1, mesh2, correspondences)

Correspondence Heatmaps
~~~~~~~~~~~~~~~~~~~~~~~

Show correspondence quality as heatmaps:

.. code-block:: python

    # Correspondence heatmap
    gfm.plot.correspondence_heatmap(mesh, correspondence_quality)

Interactive Visualization
~~~~~~~~~~~~~~~~~~~~~~~~

Interactive 3D visualization:

.. code-block:: python

    # Interactive visualization
    gfm.plot.interactive_correspondences(mesh1, mesh2, correspondences)

Performance Considerations
-------------------------

Computational Complexity
~~~~~~~~~~~~~~~~~~~~~~~~

- **Functional Maps**: :math:`O(k^3)` where :math:`k` is the number of eigenfunctions
- **Pointwise Conversion**: :math:`O(nk)` where :math:`n` is the number of vertices
- **Refinement**: :math:`O(n^2)` for dense correspondences

Memory Usage
~~~~~~~~~~~~

- **Eigenfunctions**: :math:`O(nk)` storage
- **Functional Maps**: :math:`O(k^2)` storage
- **Dense Correspondences**: :math:`O(n)` storage

Optimization Tips
~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Use fewer eigenfunctions for speed
    eigenfunctions, eigenvalues = gfm.laplacian.spectrum(mesh, k=50)
    
    # Use sparse matrices for large meshes
    L = gfm.laplacian.matrix(mesh, sparse=True)
    
    # Parallel processing for multiple shapes
    correspondences = gfm.parallel.correspondences(mesh_pairs)

For practical examples and tutorials, see the :doc:`../tutorials/correspondence` section. 