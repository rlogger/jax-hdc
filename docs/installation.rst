Installation
============

Requirements
------------

* Python 3.9+
* JAX 0.4.20+
* NumPy 1.22+
* Optax 0.1.7+

JAX-HDC is currently in **alpha** and is not yet published to PyPI.

Installing from Source
----------------------

.. code-block:: bash

   git clone https://github.com/rlogger/jax-hdc.git
   cd jax-hdc
   pip install -e .

For development (testing, linting, type checking):

.. code-block:: bash

   pip install -e ".[dev]"

For running examples (matplotlib, scikit-learn):

.. code-block:: bash

   pip install -e ".[examples]"

Using Nix
---------

For reproducible development environments:

.. code-block:: bash

   nix develop        # Enter development shell
   nix build          # Build the package
   nix run .#basic-operations
   nix run .#classification-simple

GPU/TPU
-------

JAX automatically uses GPU when available. For CUDA:

.. code-block:: bash

   pip install --upgrade "jax[cuda12]"

For TPU, install JAX with TPU support as per the `JAX installation guide <https://jax.readthedocs.io/en/latest/installation.html>`_.
