VSA Models
==========

The ``jax_hdc.vsa`` module provides Vector Symbolic Architecture model implementations.

Base Class
----------

.. autoclass:: jax_hdc.vsa.VSAModel
   :members:
   :undoc-members:

Binary Spatter Codes
--------------------

.. autoclass:: jax_hdc.vsa.BSC
   :members:
   :undoc-members:

Multiply-Add-Permute
--------------------

.. autoclass:: jax_hdc.vsa.MAP
   :members:
   :undoc-members:

Holographic Reduced Representations
-----------------------------------

.. autoclass:: jax_hdc.vsa.HRR
   :members:
   :undoc-members:

Fourier HRR
-----------

.. autoclass:: jax_hdc.vsa.FHRR
   :members:
   :undoc-members:

Factory Function
----------------

.. autofunction:: jax_hdc.vsa.create_vsa_model

Example Usage
-------------

Creating models::

   from jax_hdc import BSC, MAP, HRR, FHRR
   import jax

   key = jax.random.PRNGKey(42)

   # Binary Spatter Codes
   bsc = BSC.create(dimensions=10000)
   x = bsc.random(key, (10000,))
   y = bsc.random(key, (10000,))
   bound = bsc.bind(x, y)

   # MAP
   map_model = MAP.create(dimensions=10000)
   x = map_model.random(key, (10000,))
   y = map_model.random(key, (10000,))
   bound = map_model.bind(x, y)

Using factory function::

   from jax_hdc.vsa import create_vsa_model

   model = create_vsa_model('map', dimensions=10000)
