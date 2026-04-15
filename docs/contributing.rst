Contributing
============

We welcome contributions to JAX-HDC.

Development Setup
-----------------

1. Fork and clone the repository:

.. code-block:: bash

   git clone https://github.com/yourusername/jax-hdc.git
   cd jax-hdc

2. Create a virtual environment and install:

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate
   pip install -e ".[dev]"

Code Style
----------

We use ``ruff`` for linting and formatting:

.. code-block:: bash

   ruff check jax_hdc/ tests/
   ruff format jax_hdc/ tests/

Type checking:

.. code-block:: bash

   mypy jax_hdc/

Testing
-------

.. code-block:: bash

   pytest tests/ -v

Submitting Changes
------------------

1. Create a feature branch
2. Make changes and add tests
3. Ensure all tests pass and code is formatted
4. Submit a pull request
