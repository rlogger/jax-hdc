JAX-HDC Documentation
=====================

JAX-HDC is a high-performance library for Hyperdimensional Computing (HDC) and Vector Symbolic Architectures (VSA) built on JAX.

Features
--------

- XLA compilation and automatic kernel fusion for 10-100x speedups
- Native GPU/TPU support through JAX backend
- Pure functional design enabling JAX transformations (jit, vmap, grad, pmap)
- Four VSA model implementations: BSC, MAP, HRR, FHRR
- Feature encoders for discrete, continuous, and high-dimensional data
- Classification models with test coverage

Quick Start
-----------

Installation::

   pip install jax-hdc

Basic usage::

   import jax
   from jax_hdc import MAP

   model = MAP.create(dimensions=10000)
   key = jax.random.PRNGKey(42)

   x = model.random(key, (10000,))
   y = model.random(key, (10000,))
   bound = model.bind(x, y)
   similarity = model.similarity(x, y)

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api
   functional
   vsa
   embeddings
   models
   utils

.. toctree::
   :maxdepth: 1
   :caption: Additional Information

   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
