"""Utility functions for JAX-HDC.

This module provides helper functions for device management, memory configuration,
benchmarking, and other common operations.
"""

import os
import time
from typing import Optional, Union, List, Tuple, Callable, Any, Dict
import jax
import jax.numpy as jnp


def configure_memory(
    preallocate: bool = False,
    memory_fraction: float = 0.8,
    device: str = "gpu"
) -> None:
    """Configure JAX memory allocation settings.

    Args:
        preallocate: Whether to preallocate GPU memory (default: False).
                    False enables flexible allocation, recommended for development.
        memory_fraction: Fraction of GPU memory to use (default: 0.8)
        device: Device type ('gpu' or 'cpu')

    Example:
        >>> from jax_hdc.utils import configure_memory
        >>> # Use 90% of GPU memory with flexible allocation
        >>> configure_memory(preallocate=False, memory_fraction=0.9)
    """
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = str(preallocate).lower()
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(memory_fraction)

    if device.lower() == "gpu":
        # Enable Triton GEMM for better performance (optional)
        os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=true'


def get_device(device_type: str = "gpu", device_id: int = 0) -> jax.Device:
    """Get a JAX device.

    Args:
        device_type: Type of device ('gpu', 'cpu', or 'tpu')
        device_id: Device ID (default: 0)

    Returns:
        JAX Device object

    Example:
        >>> device = get_device('gpu', 0)
        >>> print(device)
    """
    try:
        devices = jax.devices(device_type)
        if device_id >= len(devices):
            raise ValueError(
                f"Device ID {device_id} out of range. "
                f"Available {device_type} devices: {len(devices)}"
            )
        return devices[device_id]
    except RuntimeError:
        # Fallback to CPU if requested device type not available
        print(f"Warning: {device_type} not available, falling back to CPU")
        return jax.devices('cpu')[0]


def get_device_memory_stats(device: Optional[jax.Device] = None) -> dict:
    """Get memory statistics for a device.

    Args:
        device: JAX device (default: None, uses default device)

    Returns:
        Dictionary with memory statistics

    Example:
        >>> stats = get_device_memory_stats()
        >>> print(f"Peak memory: {stats['peak_bytes_in_use'] / 1e9:.2f} GB")
    """
    if device is None:
        device = jax.devices()[0]

    try:
        stats = device.memory_stats()
        if stats is None:
            return {"error": "Memory stats not available for this device"}
        return stats
    except AttributeError:
        return {"error": "Memory stats not available for this device"}


def set_random_seed(seed: int) -> jax.Array:
    """Set random seed and return a JAX PRNG key.

    Args:
        seed: Random seed

    Returns:
        JAX PRNG key

    Example:
        >>> key = set_random_seed(42)
        >>> # Use key for random operations
        >>> x = jax.random.normal(key, (100,))
    """
    return jax.random.PRNGKey(seed)


def benchmark_function(
    fn: Callable[..., Any],
    *args: Any,
    num_trials: int = 100,
    warmup: int = 10,
    **kwargs: Any
) -> Dict[str, Union[float, int]]:
    """Benchmark a JAX function.

    Properly handles JIT compilation and async dispatch for accurate timing.

    Args:
        fn: Function to benchmark
        *args: Positional arguments to fn
        num_trials: Number of trials to run (default: 100)
        warmup: Number of warmup trials (default: 10)
        **kwargs: Keyword arguments to fn

    Returns:
        Dictionary with timing statistics (mean, std, min, max in milliseconds)

    Example:
        >>> import jax.numpy as jnp
        >>> from jax_hdc.functional import bind_map
        >>> x = jnp.ones(10000)
        >>> y = jnp.ones(10000)
        >>> stats = benchmark_function(bind_map, x, y, num_trials=100)
        >>> print(f"Mean time: {stats['mean_ms']:.3f} ms")
    """
    # Warmup (includes compilation)
    for _ in range(warmup):
        result = fn(*args, **kwargs)
        jax.block_until_ready(result)

    # Benchmark
    times_list: List[float] = []
    for _ in range(num_trials):
        start = time.time()
        result = fn(*args, **kwargs)
        jax.block_until_ready(result)  # Wait for async dispatch
        end = time.time()
        times_list.append((end - start) * 1000)  # Convert to ms

    times = jnp.array(times_list)

    return {
        'mean_ms': float(jnp.mean(times)),
        'std_ms': float(jnp.std(times)),
        'min_ms': float(jnp.min(times)),
        'max_ms': float(jnp.max(times)),
        'median_ms': float(jnp.median(times)),
        'num_trials': num_trials
    }


def check_shapes(*arrays: jax.Array, expected_ndim: Optional[int] = None) -> None:
    """Validate array shapes for debugging.

    Args:
        *arrays: Arrays to check
        expected_ndim: Expected number of dimensions (default: None)

    Raises:
        ValueError: If shapes are inconsistent or don't match expected_ndim

    Example:
        >>> x = jnp.ones((100, 10000))
        >>> y = jnp.ones((100, 10000))
        >>> check_shapes(x, y, expected_ndim=2)  # OK
        >>> z = jnp.ones((50, 10000))
        >>> check_shapes(x, z)  # Raises ValueError
    """
    if len(arrays) == 0:
        return

    shapes = [arr.shape for arr in arrays]
    ndims = [arr.ndim for arr in arrays]

    # Check ndim
    if expected_ndim is not None:
        for i, ndim in enumerate(ndims):
            if ndim != expected_ndim:
                raise ValueError(
                    f"Array {i} has {ndim} dimensions, expected {expected_ndim}"
                )

    # Check shapes are compatible
    first_shape = shapes[0]
    for i, shape in enumerate(shapes[1:], 1):
        if shape != first_shape:
            raise ValueError(
                f"Shape mismatch: array 0 has shape {first_shape}, "
                f"array {i} has shape {shape}"
            )


def normalize(x: jax.Array, axis: int = -1, eps: float = 1e-8) -> jax.Array:
    """Normalize vectors to unit length.

    Args:
        x: Input array
        axis: Axis along which to normalize (default: -1)
        eps: Small constant to avoid division by zero (default: 1e-8)

    Returns:
        Normalized array

    Example:
        >>> x = jax.random.normal(jax.random.PRNGKey(0), (100, 10000))
        >>> x_norm = normalize(x)
        >>> # Check that vectors have unit norm
        >>> norms = jnp.linalg.norm(x_norm, axis=-1)
        >>> assert jnp.allclose(norms, 1.0)
    """
    norm = jnp.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + eps)


def print_model_info(model: Any) -> None:
    """Print information about a VSA model or encoder.

    Args:
        model: VSA model, encoder, or classifier instance

    Example:
        >>> from jax_hdc import MAP
        >>> model = MAP.create(dimensions=10000)
        >>> print_model_info(model)
    """
    print(f"Model: {model.__class__.__name__}")

    # Print all fields
    if hasattr(model, '__dataclass_fields__'):
        for field_name in model.__dataclass_fields__:
            value = getattr(model, field_name)

            # Handle arrays specially
            if isinstance(value, jax.Array):
                print(f"  {field_name}: Array{value.shape} dtype={value.dtype}")
            else:
                print(f"  {field_name}: {value}")


def count_parameters(model: Any) -> int:
    """Count total number of parameters in a model.

    Args:
        model: Model instance (VSA, encoder, or classifier)

    Returns:
        Total number of parameters

    Example:
        >>> from jax_hdc import MAP, RandomEncoder
        >>> encoder = RandomEncoder.create(100, 100, 10000, 'map')
        >>> num_params = count_parameters(encoder)
        >>> print(f"Total parameters: {num_params:,}")
    """
    total = 0

    if hasattr(model, '__dataclass_fields__'):
        for field_name in model.__dataclass_fields__:
            value = getattr(model, field_name)
            if isinstance(value, jax.Array):
                total += value.size

    return total


def to_device(data: Union[jax.Array, Tuple[Any, ...], List[Any]], device: jax.Device) -> Union[jax.Array, Tuple[Any, ...], List[Any]]:
    """Move data to a specific device.

    Args:
        data: Data to move (array, tuple, or list of arrays)
        device: Target device

    Returns:
        Data on the target device

    Example:
        >>> device = get_device('gpu', 0)
        >>> x = jnp.ones(10000)
        >>> x_gpu = to_device(x, device)
    """
    if isinstance(data, jax.Array):
        return jax.device_put(data, device)
    elif isinstance(data, (tuple, list)):
        result: Union[Tuple[Any, ...], List[Any]] = type(data)(jax.device_put(item, device) for item in data)
        return result
    return data


def check_nan_inf(x: jax.Array, name: str = "array") -> None:
    """Check for NaN or Inf values in an array.

    Args:
        x: Array to check
        name: Name for error messages (default: "array")

    Raises:
        ValueError: If NaN or Inf values are found

    Example:
        >>> x = jnp.array([1.0, 2.0, 3.0])
        >>> check_nan_inf(x)  # OK
        >>> y = jnp.array([1.0, jnp.nan, 3.0])
        >>> check_nan_inf(y)  # Raises ValueError
    """
    if jnp.any(jnp.isnan(x)):
        raise ValueError(f"{name} contains NaN values")
    if jnp.any(jnp.isinf(x)):
        raise ValueError(f"{name} contains Inf values")


def get_version_info() -> dict:
    """Get version information for JAX-HDC and dependencies.

    Returns:
        Dictionary with version information

    Example:
        >>> info = get_version_info()
        >>> print(info['jax_hdc'])
        >>> print(info['jax'])
    """
    import jax_hdc

    info = {
        'jax_hdc': jax_hdc.__version__,
        'jax': jax.__version__,
    }

    try:
        import jaxlib
        info['jaxlib'] = jaxlib.__version__
    except ImportError:
        info['jaxlib'] = 'not installed'

    try:
        import optax
        info['optax'] = optax.__version__
    except ImportError:
        info['optax'] = 'not installed'

    return info


__all__ = [
    "configure_memory",
    "get_device",
    "get_device_memory_stats",
    "set_random_seed",
    "benchmark_function",
    "check_shapes",
    "normalize",
    "print_model_info",
    "count_parameters",
    "to_device",
    "check_nan_inf",
    "get_version_info",
]
