Embeddings Module
=================

The ``jax_hdc.embeddings`` module provides encoders for transforming data into hypervectors.

RandomEncoder
-------------

.. autoclass:: jax_hdc.embeddings.RandomEncoder
   :members:
   :undoc-members:

Example::

   from jax_hdc import MAP, RandomEncoder
   import jax

   model = MAP.create(dimensions=10000)
   key = jax.random.PRNGKey(42)

   encoder = RandomEncoder.create(
       num_features=20,
       num_values=10,
       dimensions=10000,
       vsa_model=model,
       key=key
   )

   # Encode discrete features
   data = jax.random.randint(key, (20,), 0, 10)
   encoded = encoder.encode(data)

LevelEncoder
------------

.. autoclass:: jax_hdc.embeddings.LevelEncoder
   :members:
   :undoc-members:

Example::

   from jax_hdc import LevelEncoder

   encoder = LevelEncoder.create(
       num_levels=100,
       dimensions=10000,
       min_value=0.0,
       max_value=1.0,
       vsa_model=model,
       key=key
   )

   # Encode continuous value
   encoded = encoder.encode(0.75)

ProjectionEncoder
-----------------

.. autoclass:: jax_hdc.embeddings.ProjectionEncoder
   :members:
   :undoc-members:

Example::

   from jax_hdc import ProjectionEncoder

   encoder = ProjectionEncoder.create(
       input_dim=784,
       dimensions=10000,
       vsa_model=model,
       key=key
   )

   # Encode high-dimensional input
   image = jax.random.normal(key, (784,))
   encoded = encoder.encode(image)
