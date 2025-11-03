Utilities Module
================

The ``jax_hdc.utils`` module provides utility functions for configuration and benchmarking.

Configuration
-------------

.. autofunction:: jax_hdc.utils.configure_memory

.. autofunction:: jax_hdc.utils.get_device

.. autofunction:: jax_hdc.utils.get_device_memory_stats

Random Number Generation
-------------------------

.. autofunction:: jax_hdc.utils.set_random_seed

Benchmarking
------------

.. autofunction:: jax_hdc.utils.benchmark_function

Validation
----------

.. autofunction:: jax_hdc.utils.check_shapes

.. autofunction:: jax_hdc.utils.check_nan_inf

Helpers
-------

.. autofunction:: jax_hdc.utils.normalize

.. autofunction:: jax_hdc.utils.print_model_info

.. autofunction:: jax_hdc.utils.count_parameters

.. autofunction:: jax_hdc.utils.to_device

.. autofunction:: jax_hdc.utils.get_version_info

Example Usage
-------------

Memory configuration::

   from jax_hdc.utils import configure_memory

   # Use 90% of GPU memory with flexible allocation
   configure_memory(preallocate=False, memory_fraction=0.9)

Device management::

   from jax_hdc.utils import get_device, to_device

   device = get_device('gpu', 0)
   data_on_gpu = to_device(data, device)

Benchmarking::

   from jax_hdc.utils import benchmark_function
   from jax_hdc.functional import bind_map

   stats = benchmark_function(bind_map, x, y, num_trials=100)
   print(f"Mean time: {stats['mean_ms']:.3f} ms")
