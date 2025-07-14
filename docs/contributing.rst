Contributing to GeomFuM
=======================

We welcome contributions from the community! This guide will help you get started with contributing to GeomFuM.

Getting Started
---------------

Development Setup
~~~~~~~~~~~~~~~~~

1. **Fork the repository**:
   Visit `https://github.com/DiG-AIR/geomfum` and click "Fork"

2. **Clone your fork**:
   .. code-block:: bash

       git clone https://github.com/YOUR_USERNAME/geomfum.git
       cd geomfum

3. **Install in development mode**:
   .. code-block:: bash

       pip install -e ".[test,test-scripts,plotting-all]"

4. **Set up pre-commit hooks**:
   .. code-block:: bash

       pip install pre-commit
       pre-commit install

Development Environment
~~~~~~~~~~~~~~~~~~~~~~~

We recommend using a virtual environment:

.. code-block:: bash

    # Create virtual environment
    python -m venv geomfum_env
    source geomfum_env/bin/activate  # On Windows: geomfum_env\Scripts\activate
    
    # Install development dependencies
    pip install -e ".[test,test-scripts,plotting-all]"

Code Style
----------

We follow PEP 8 and use several tools to maintain code quality:

Formatting
~~~~~~~~~~

We use `ruff` for code formatting and linting:

.. code-block:: bash

    # Format code
    ruff format .
    
    # Check for issues
    ruff check .
    
    # Fix issues automatically
    ruff check --fix .

Type Hints
~~~~~~~~~~

We use type hints throughout the codebase:

.. code-block:: python

    from typing import Optional, Tuple, List
    import numpy as np
    
    def compute_functional_map(
        mesh1: Mesh,
        mesh2: Mesh,
        descriptors1: np.ndarray,
        descriptors2: np.ndarray,
        k: int = 50
    ) -> np.ndarray:
        """Compute functional map between two shapes.
        
        Args:
            mesh1: First mesh
            mesh2: Second mesh
            descriptors1: Descriptors for first mesh
            descriptors2: Descriptors for second mesh
            k: Number of eigenfunctions to use
            
        Returns:
            Functional map matrix
        """
        # Implementation here
        pass

Documentation
~~~~~~~~~~~~~

We use NumPy docstring format:

.. code-block:: python

    def laplacian_spectrum(mesh: Mesh, k: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Laplace-Beltrami eigenfunctions and eigenvalues.
        
        Parameters
        ----------
        mesh : Mesh
            Input mesh
        k : int, optional
            Number of eigenfunctions to compute, by default 50
            
        Returns
        -------
        eigenfunctions : np.ndarray
            Eigenfunctions matrix of shape (n_vertices, k)
        eigenvalues : np.ndarray
            Eigenvalues array of length k
            
        Notes
        -----
        The eigenfunctions are computed using the cotangent Laplacian.
        """
        pass

Testing
-------

Running Tests
~~~~~~~~~~~~~

Run the test suite:

.. code-block:: bash

    # Run all tests
    pytest
    
    # Run specific test file
    pytest tests/test_functional_map.py
    
    # Run with coverage
    pytest --cov=geomfum
    
    # Run specific test categories
    pytest -m "smoke"  # Simple tests
    pytest -m "random"  # Randomized tests
    pytest -m "validation"  # Validation tests

Writing Tests
~~~~~~~~~~~~

Follow these guidelines for writing tests:

1. **Test structure**:
   .. code-block:: python

       def test_functional_map_basic():
           """Test basic functional map computation."""
           # Arrange
           mesh1 = create_test_mesh()
           mesh2 = create_test_mesh()
           descriptors1 = np.random.rand(100, 3)
           descriptors2 = np.random.rand(100, 3)
           
           # Act
           result = gfm.functional_map(mesh1, mesh2, descriptors1, descriptors2)
           
           # Assert
           assert result.shape == (50, 50)
           assert np.all(np.isfinite(result))

2. **Test categories**:
   - `@pytest.mark.smoke`: Simple and basic numerical tests
   - `@pytest.mark.random`: Tests that use randomized data
   - `@pytest.mark.validation`: Not smoke, neither random
   - `@pytest.mark.slow`: For slow tests

3. **Fixtures**:
   .. code-block:: python

       @pytest.fixture
       def sample_mesh():
           """Create a sample mesh for testing."""
           vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
           faces = np.array([[0, 1, 2]])
           return gfm.shape.Mesh(vertices, faces)

Documentation
-------------

Building Documentation
~~~~~~~~~~~~~~~~~~~~~

Build the documentation locally:

.. code-block:: bash

    # Install documentation dependencies
    pip install -r docs/requirements.txt
    
    # Build documentation
    cd docs
    make html
    
    # View documentation
    open _build/html/index.html

Writing Documentation
~~~~~~~~~~~~~~~~~~~~

1. **API Documentation**: Add docstrings to all public functions
2. **Tutorials**: Create Jupyter notebooks in `notebooks/`
3. **Concept Documentation**: Add theoretical explanations in `docs/concepts/`

Example docstring:

.. code-block:: python

    def zoomout_refinement(
        functional_map: np.ndarray,
        mesh1: Mesh,
        mesh2: Mesh,
        iterations: int = 5
    ) -> np.ndarray:
        """Apply ZoomOut refinement to functional map.
        
        ZoomOut is a spectral upsampling technique that improves
        correspondence quality by iteratively refining the functional map.
        
        Parameters
        ----------
        functional_map : np.ndarray
            Input functional map matrix
        mesh1 : Mesh
            First mesh
        mesh2 : Mesh
            Second mesh
        iterations : int, optional
            Number of refinement iterations, by default 5
            
        Returns
        -------
        np.ndarray
            Refined functional map
            
        References
        ----------
        .. [1] Melzi, S., et al. "ZoomOut: Spectral Upsampling for Efficient
               Shape Correspondence." ACM TOG 38.6 (2019): 155.
        """
        pass

Pull Request Process
-------------------

1. **Create a feature branch**:
   .. code-block:: bash

       git checkout -b feature/your-feature-name

2. **Make your changes**:
   - Write code following our style guidelines
   - Add tests for new functionality
   - Update documentation
   - Update CHANGELOG.md if needed

3. **Run tests and checks**:
   .. code-block:: bash

       # Run tests
       pytest
       
       # Check code style
       ruff check .
       
       # Build documentation
       cd docs && make html

4. **Commit your changes**:
   .. code-block:: bash

       git add .
       git commit -m "Add feature: brief description"
       
       # Use conventional commit format:
       # feat: add new functional map refinement
       # fix: resolve memory leak in laplacian computation
       # docs: update installation instructions

5. **Push and create PR**:
   .. code-block:: bash

       git push origin feature/your-feature-name

6. **Create Pull Request**:
   - Use the PR template
   - Describe your changes clearly
   - Link related issues
   - Request reviews from maintainers

Code Review
-----------

Review Process
~~~~~~~~~~~~~

1. **Automated checks** must pass:
   - Tests
   - Code style
   - Documentation build
   - Type checking

2. **Manual review** by maintainers:
   - Code quality
   - Algorithm correctness
   - Performance considerations
   - Documentation quality

3. **Address feedback**:
   - Respond to review comments
   - Make requested changes
   - Re-request review when ready

Review Guidelines
~~~~~~~~~~~~~~~~~

When reviewing code:

1. **Check correctness**: Is the algorithm implemented correctly?
2. **Check performance**: Are there performance issues?
3. **Check style**: Does the code follow our conventions?
4. **Check documentation**: Is the documentation clear and complete?
5. **Check tests**: Are there adequate tests?

Areas for Contribution
----------------------

We welcome contributions in these areas:

Algorithms
~~~~~~~~~~

- New functional map algorithms
- Improved refinement techniques
- Novel shape descriptors
- Deep learning approaches

Performance
~~~~~~~~~~~

- GPU acceleration
- Parallel processing
- Memory optimization
- Sparse matrix operations

Documentation
~~~~~~~~~~~~~

- Tutorials and examples
- API documentation
- Theoretical explanations
- Performance benchmarks

Testing
~~~~~~~

- Unit tests
- Integration tests
- Performance tests
- Documentation tests

Tools and Utilities
~~~~~~~~~~~~~~~~~~

- Visualization tools
- Data processing utilities
- Evaluation metrics
- Benchmarking tools

Getting Help
-----------

If you need help:

1. **Check existing issues**: Search for similar problems
2. **Join Discord**: `https://discord.gg/6sYmEbUp`
3. **Create an issue**: For bugs or feature requests
4. **Ask questions**: In GitHub Discussions

Community Guidelines
-------------------

We are committed to providing a welcoming and inclusive environment:

1. **Be respectful**: Treat everyone with respect
2. **Be helpful**: Help others learn and contribute
3. **Be patient**: Everyone learns at their own pace
4. **Be constructive**: Provide constructive feedback

Code of Conduct
~~~~~~~~~~~~~~~

We follow the Contributor Covenant Code of Conduct. Please read it at:
`https://www.contributor-covenant.org/version/2/0/code_of_conduct.html`

Recognition
-----------

Contributors are recognized in:

1. **GitHub contributors list**
2. **Documentation acknowledgments**
3. **Release notes**
4. **Academic citations** (for significant contributions)

Thank you for contributing to GeomFuM! 🎉