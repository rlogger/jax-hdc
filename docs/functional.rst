Functional Module
=================

The ``jax_hdc.functional`` module provides core HDC operations as pure functions.

BSC Operations
--------------

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

CGR Operations
--------------

.. autofunction:: jax_hdc.functional.bind_cgr
.. autofunction:: jax_hdc.functional.bundle_cgr
.. autofunction:: jax_hdc.functional.inverse_cgr
.. autofunction:: jax_hdc.functional.matching_similarity

MCR Operations
--------------

.. autofunction:: jax_hdc.functional.bind_mcr
.. autofunction:: jax_hdc.functional.bundle_mcr
.. autofunction:: jax_hdc.functional.inverse_mcr
.. autofunction:: jax_hdc.functional.phasor_similarity

VTB Operations
--------------

.. autofunction:: jax_hdc.functional.bind_vtb
.. autofunction:: jax_hdc.functional.bundle_vtb
.. autofunction:: jax_hdc.functional.inverse_vtb

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
