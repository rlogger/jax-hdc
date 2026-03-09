VSA Models
==========

The ``jax_hdc.vsa`` module provides Vector Symbolic Architecture model implementations.

All models share the same API: ``bind``, ``bundle``, ``inverse``, ``similarity``, ``random``.

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

Binary Sparse Block Codes
--------------------------

.. autoclass:: jax_hdc.vsa.BSBC
   :members:
   :undoc-members:

Cyclic Group Representation
----------------------------

.. autoclass:: jax_hdc.vsa.CGR
   :members:
   :undoc-members:

Modular Composite Representation
---------------------------------

.. autoclass:: jax_hdc.vsa.MCR
   :members:
   :undoc-members:

Vector-Derived Transformation Binding
--------------------------------------

.. autoclass:: jax_hdc.vsa.VTB
   :members:
   :undoc-members:

Factory Function
----------------

.. autofunction:: jax_hdc.vsa.create_vsa_model
