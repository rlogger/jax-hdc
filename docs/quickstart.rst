Quick Start
===========

This guide helps you get started with JAX-HDC quickly.

Basic Usage
-----------

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from jax_hdc import MAP

   model = MAP.create(dimensions=10000)
   key = jax.random.PRNGKey(42)

   # Generate random hypervectors
   x = model.random(key, (10000,))
   y = model.random(jax.random.split(key)[1], (10000,))

   # Bind and bundle
   bound = model.bind(x, y)
   bundled = model.bundle(jnp.stack([x, y]), axis=0)

   # Similarity
   sim = model.similarity(x, y)

Classification Pipeline
-----------------------

.. code-block:: python

   from jax_hdc import MAP, RandomEncoder, CentroidClassifier

   model = MAP.create(dimensions=10000)
   key = jax.random.PRNGKey(42)

   # Create encoder and classifier
   encoder = RandomEncoder.create(
       num_features=20, num_values=10, dimensions=10000,
       vsa_model=model, key=key
   )
   classifier = CentroidClassifier.create(
       num_classes=5, dimensions=10000, vsa_model=model
   )

   # Encode and train
   data = jax.random.randint(key, (100, 20), 0, 10)
   labels = jax.random.randint(key, (100,), 0, 5)
   encoded = encoder.encode_batch(data)
   classifier = classifier.fit(encoded, labels)

   # Predict and score
   predictions = classifier.predict(encoded)
   accuracy = classifier.score(encoded, labels)
   print(f"Accuracy: {accuracy:.2%}")

Key Concepts
------------

Hyperdimensional Computing (HDC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

HDC uses high-dimensional vectors to represent and manipulate information:

* **Hypervectors**: High-dimensional vectors (typically 1000+ dimensions)
* **Binding**: Combines two hypervectors into a dissimilar result
* **Bundling**: Superposes multiple hypervectors into a similar result
* **Similarity**: Measures relatedness between hypervectors

JAX Integration
~~~~~~~~~~~~~~~

JAX-HDC leverages JAX for:

* **JIT compilation**: Fast execution of HDC operations
* **vmap**: Efficient batch processing
* **Hardware acceleration**: GPU/TPU support through JAX
* **Functional design**: Works with jit, vmap, pmap, grad
