Functional Module
=================

The ``jax_hdc.functional`` module provides core HDC operations implemented as pure functions.

Binary Spatter Code Operations
-------------------------------

.. autofunction:: jax_hdc.functional.bind_bsc

.. autofunction:: jax_hdc.functional.bundle_bsc

.. autofunction:: jax_hdc.functional.inverse_bsc

.. autofunction:: jax_hdc.functional.hamming_similarity

MAP Operations
--------------

.. autofunction:: jax_hdc.functional.bind_map

.. autofunction:: jax_hdc.functional.bundle_map

.. autofunction:: jax_hdc.functional.inverse_map

.. autofunction:: jax_hdc.functional.cosine_similarity

HRR Operations
--------------

.. autofunction:: jax_hdc.functional.bind_hrr

.. autofunction:: jax_hdc.functional.bundle_hrr

.. autofunction:: jax_hdc.functional.inverse_hrr

Universal Operations
--------------------

.. autofunction:: jax_hdc.functional.permute

.. autofunction:: jax_hdc.functional.cleanup

Batch Operations
----------------

.. autofunction:: jax_hdc.functional.batch_bind_bsc

.. autofunction:: jax_hdc.functional.batch_bind_map

.. autofunction:: jax_hdc.functional.batch_hamming_similarity

.. autofunction:: jax_hdc.functional.batch_cosine_similarity

Example Usage
-------------

Basic binding and bundling::

   import jax
   import jax.numpy as jnp
   from jax_hdc import functional as F

   key = jax.random.PRNGKey(42)

   # BSC operations
   x = jax.random.bernoulli(key, 0.5, shape=(10000,))
   y = jax.random.bernoulli(key, 0.5, shape=(10000,))

   bound = F.bind_bsc(x, y)
   sim = F.hamming_similarity(x, y)

   # MAP operations
   x = jax.random.normal(key, shape=(10000,))
   y = jax.random.normal(key, shape=(10000,))

   bound = F.bind_map(x, y)
   sim = F.cosine_similarity(x, y)

Batch processing::

   vectors = jax.random.normal(key, shape=(100, 10000))
   bundled = F.bundle_map(vectors, axis=0)
