Usage
=====

This page provides comprehensive usage examples for GeomFuM, referencing our interactive notebook tutorials.

Quick Start
-----------

The best way to learn GeomFuM is through our interactive notebook tutorials. Start with:

- **Loading Meshes** - :doc:`notebooks/how_to/00_load_mesh_from_file`
- **Laplace-Beltrami Operator** - :doc:`notebooks/how_to/01_mesh_laplacian`
- **Functional Maps** - :doc:`notebooks/how_to/07_functional_map`

Basic Usage Examples
-------------------

Here are some key code snippets from our tutorials:

Loading and Working with Meshes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From :doc:`notebooks/how_to/00_load_mesh_from_file`:

.. code-block:: python

    from geomfum.dataset import NotebooksDataset
    from geomfum.shape import TriangleMesh, PointCloud
    
    # Load a mesh
    dataset = NotebooksDataset()
    mesh = TriangleMesh.from_file(dataset.get_filename("cat-00"))
    
    print(f"Mesh: {mesh.n_vertices} vertices, {mesh.n_faces} faces")

Computing Laplace-Beltrami Spectrum
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From :doc:`notebooks/how_to/02_mesh_laplacian_spectrum`:

.. code-block:: python

    # Compute eigenfunctions and eigenvalues
    mesh.laplacian.find_spectrum(spectrum_size=50, set_as_basis=True)
    
    # Access the basis
    eigenfunctions = mesh.basis.eigenfunctions
    eigenvalues = mesh.basis.eigenvalues
    
    print(f"Computed {len(eigenvalues)} eigenfunctions")

Computing Shape Descriptors
~~~~~~~~~~~~~~~~~~~~~~~~~~~

From :doc:`notebooks/how_to/03_descriptors`:

.. code-block:: python

    from geomfum.descriptor.spectral import HeatKernelSignature, WaveKernelSignature
    
    # Heat Kernel Signature
    hks = HeatKernelSignature.from_registry(n_domain=4)
    hks_descriptors = hks.apply(mesh)
    
    # Wave Kernel Signature
    wks = WaveKernelSignature.from_registry(n_domain=3)
    wks_descriptors = wks.apply(mesh)

Creating Descriptor Pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From :doc:`notebooks/how_to/04_descriptor_pipeline`:

.. code-block:: python

    from geomfum.descriptor.pipeline import DescriptorPipeline, ArangeSubsampler, L2InnerNormalizer
    
    steps = [
        HeatKernelSignature.from_registry(n_domain=4),
        ArangeSubsampler(subsample_step=2),
        WaveKernelSignature.from_registry(n_domain=3),
        L2InnerNormalizer(),
    ]
    
    pipeline = DescriptorPipeline(steps)
    descriptors = pipeline.apply(mesh)

Computing Functional Maps
------------------------

From :doc:`notebooks/how_to/07_functional_map`:

.. code-block:: python

    from geomfum.functional_map import (
        FactorSum, LBCommutativityEnforcing, 
        SpectralDescriptorPreservation, OperatorCommutativityEnforcing
    )
    from geomfum.numerics.optimization import ScipyMinimize
    
    # Set up basis for both meshes
    mesh_a.laplacian.find_spectrum(spectrum_size=10, set_as_basis=True)
    mesh_b.laplacian.find_spectrum(spectrum_size=10, set_as_basis=True)
    
    # Compute descriptors
    descr_a = pipeline.apply(mesh_a)
    descr_b = pipeline.apply(mesh_b)
    
    # Create objective function
    factors = [
        SpectralDescriptorPreservation(
            mesh_a.basis.project(descr_a),
            mesh_b.basis.project(descr_b),
            weight=1.0,
        ),
        LBCommutativityEnforcing.from_bases(
            mesh_a.basis, mesh_b.basis, weight=1e-2,
        ),
    ]
    
    objective = FactorSum(factors)
    
    # Optimize
    optimizer = ScipyMinimize(method="L-BFGS-B")
    x0 = gs.zeros((mesh_b.basis.spectrum_size, mesh_a.basis.spectrum_size))
    res = optimizer.minimize(objective, x0, fun_jac=objective.gradient)
    fmap = res.x.reshape(x0.shape)

Converting to Pointwise Correspondences
--------------------------------------

From :doc:`notebooks/how_to/10_pointwise_from_functional`:

.. code-block:: python

    # Convert functional map to pointwise correspondences
    correspondences = mesh_a.basis.pointwise_from_functional(fmap, mesh_b.basis)
    
    print(f"Computed {len(correspondences)} correspondences")

Refining Functional Maps
-----------------------

From :doc:`notebooks/how_to/15_refine_functional_map`:

.. code-block:: python

    from geomfum.refine import ZoomOut
    
    # Apply ZoomOut refinement
    zoomout = ZoomOut()
    refined_fmap = zoomout.apply(fmap, mesh_a, mesh_b)
    
    # Convert refined map to correspondences
    refined_correspondences = mesh_a.basis.pointwise_from_functional(refined_fmap, mesh_b.basis)

Advanced Techniques
------------------

Landmark-Based Functional Maps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From :doc:`notebooks/how_to/06_landmarks`:

.. code-block:: python

    # Define landmark correspondences
    landmarks_a = [0, 10, 20]  # Vertex indices on mesh_a
    landmarks_b = [5, 15, 25]  # Corresponding vertices on mesh_b
    
    # Create landmark-based functional map
    landmark_fmap = create_landmark_functional_map(
        mesh_a, mesh_b, landmarks_a, landmarks_b
    )

Hierarchical Meshes
~~~~~~~~~~~~~~~~~~

From :doc:`notebooks/how_to/11_hierarchical_mesh`:

.. code-block:: python

    from geomfum.shape import HierarchicalMesh
    
    # Create hierarchical mesh
    hierarchical_mesh = HierarchicalMesh(mesh, levels=3)
    
    # Work with different levels
    for level in range(hierarchical_mesh.num_levels):
        level_mesh = hierarchical_mesh.get_level(level)
        # Process each level...

Rematching Algorithm
~~~~~~~~~~~~~~~~~~~

From :doc:`notebooks/how_to/13_rematching`:

.. code-block:: python

    from geomfum.rematching import Rematching
    
    # Apply Rematching
    rematching = Rematching()
    rematched_fmap = rematching.apply(fmap, mesh_a, mesh_b)

Deep Learning Integration
------------------------

Deep Functional Maps
~~~~~~~~~~~~~~~~~~~

From :doc:`notebooks/how_to/14_deep_functional_maps_models`:

.. code-block:: python

    from geomfum.learning import DeepFunctionalMaps
    
    # Create deep functional maps model
    model = DeepFunctionalMaps(feature_dim=128)
    
    # Extract deep features
    features_a = model.extract_features(mesh_a)
    features_b = model.extract_features(mesh_b)
    
    # Compute functional map with deep features
    deep_fmap = compute_functional_map(features_a, features_b)

Neural Adjoint Maps
~~~~~~~~~~~~~~~~~~

From :doc:`notebooks/how_to/18_neural_adjoint_maps`:

.. code-block:: python

    from geomfum.learning import NeuralAdjointMap
    
    # Create neural adjoint map model
    model = NeuralAdjointMap()
    
    # Compute correspondences
    correspondences = model(mesh_a, mesh_b)

Visualization
-------------

Basic Visualization
~~~~~~~~~~~~~~~~~~

From :doc:`notebooks/how_to/16_vis_basic`:

.. code-block:: python

    # Visualize meshes and correspondences
    plot_mesh(mesh_a, title="Mesh A")
    plot_mesh(mesh_b, title="Mesh B")
    plot_correspondences(mesh_a, mesh_b, correspondences[:100])

Distance Visualization
~~~~~~~~~~~~~~~~~~~~~

From :doc:`notebooks/how_to/17_vis_dist`:

.. code-block:: python

    # Visualize geodesic distances and errors
    plot_geodesic_distances(mesh, distances)
    plot_correspondence_errors(mesh_a, mesh_b, correspondences, ground_truth)

Complete Pipeline Example
------------------------

Here's a complete example combining multiple techniques:

.. code-block:: python

    # 1. Load meshes
    mesh_a = TriangleMesh.from_file(dataset.get_filename("cat-00"))
    mesh_b = TriangleMesh.from_file(dataset.get_filename("lion-00"))
    
    # 2. Set up basis
    mesh_a.laplacian.find_spectrum(spectrum_size=50, set_as_basis=True)
    mesh_b.laplacian.find_spectrum(spectrum_size=50, set_as_basis=True)
    
    # 3. Compute descriptors
    pipeline = create_descriptor_pipeline()
    descr_a = pipeline.apply(mesh_a)
    descr_b = pipeline.apply(mesh_b)
    
    # 4. Compute functional map
    fmap = compute_functional_map(descr_a, descr_b, mesh_a, mesh_b)
    
    # 5. Refine
    refined_fmap = ZoomOut().apply(fmap, mesh_a, mesh_b)
    
    # 6. Convert to correspondences
    correspondences = mesh_a.basis.pointwise_from_functional(refined_fmap, mesh_b.basis)
    
    # 7. Visualize
    plot_correspondences(mesh_a, mesh_b, correspondences[:100])

Interactive Learning
-------------------

For the best learning experience:

1. **Follow the notebook tutorials** in order
2. **Run each cell** to see the outputs
3. **Modify parameters** to experiment
4. **Try your own data** by changing file paths
5. **Use the links** at the end of each notebook to navigate

All examples in this page are taken from our actual working notebooks. For complete, runnable examples with outputs and visualizations, see the :doc:`notebooks/index` section.

Next Steps
----------

Now that you understand the basic usage patterns:

1. **Explore the notebooks** - Follow the interactive tutorials
2. **Read the concepts** - Understand the theoretical foundations
3. **Check the API** - Reference the complete API documentation
4. **Join the community** - Visit our Discord server

For questions and issues, visit our `GitHub repository <https://github.com/DiG-AIR/geomfum>`_.