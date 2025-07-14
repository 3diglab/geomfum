.. image:: GeomFuMlogo.png
   :width: 800
   :alt: GeomFuM logo

Welcome to GeomFuM
==================

**GeomFuM** is a Modular Python Package for Machine Learning with `Functional Maps <https://dl.acm.org/doi/10.1145/2185520.2185526>`_.

Functional Maps provide a powerful framework for representing and computing correspondences between geometric shapes. GeomFuM offers a comprehensive suite of tools for:

- **Shape Analysis**: Laplace-Beltrami eigenfunctions, heat kernel signatures, wave kernel signatures
- **Correspondence Computation**: Functional maps, pointwise correspondences, refinement algorithms
- **Deep Learning Integration**: Neural functional maps, feature learning for shape correspondence
- **Geometric Processing**: Mesh operations, hierarchical representations, sampling strategies

Installation
------------

.. code-block:: bash

    pip install geomfum@git+https://github.com/DiG-AIR/geomfum.git@main

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   installation
   usage
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: Core Concepts:

   concepts/functional_maps
   concepts/shape_analysis
   concepts/correspondence

.. toctree::
   :maxdepth: 2
   :caption: Interactive Tutorials:

   notebooks/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Contributing:

   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`