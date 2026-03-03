Examples
========

JAX-HDC includes runnable examples in the ``examples/`` directory.

Basic Operations
----------------

Core HDC operations: binding, bundling, permutation, and similarity.

.. code-block:: bash

   python examples/basic_operations.py

.. code-block:: python

   from jax_hdc import MAP
   import jax

   model = MAP.create(dimensions=10000)
   key = jax.random.PRNGKey(42)
   x, y = model.random(key, (2, 10000))

   bound = model.bind(x, y)
   bundled = model.bundle(jnp.stack([x, y]), axis=0)
   sim = model.similarity(x, y)

Classification
--------------

End-to-end classification with synthetic data: encode → train → evaluate.

.. code-block:: bash

   pip install -e ".[examples]"   # optional: for matplotlib, scikit-learn
   python examples/classification_simple.py

.. code-block:: python

   from jax_hdc import MAP, RandomEncoder, CentroidClassifier

   model = MAP.create(dimensions=10000)
   encoder = RandomEncoder.create(
       num_features=20, num_values=10, dimensions=10000,
       vsa_model=model, key=jax.random.PRNGKey(42)
   )
   classifier = CentroidClassifier.create(
       num_classes=5, dimensions=10000, vsa_model=model
   )

   # Encode, fit, predict
   key = jax.random.PRNGKey(42)
   data = jax.random.randint(key, (100, 20), 0, 10)
   labels = jax.random.randint(key, (100,), 0, 5)
   encoded = encoder.encode_batch(data)
   classifier = classifier.fit(encoded, labels)
   predictions = classifier.predict(encoded)
   accuracy = classifier.score(encoded, labels)

Kanerva's "Dollar of Mexico"
----------------------------

Structured knowledge representation and analogical reasoning.

.. code-block:: bash

   python examples/kanerva_example.py

See the :doc:`classification` tutorial for a step-by-step walkthrough of the classification pipeline.
