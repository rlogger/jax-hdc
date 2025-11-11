"""Core functional operations for Hyperdimensional Computing.

This module provides the fundamental operations for manipulating hypervectors:
- Binding: Combines two hypervectors into a dissimilar result
- Bundling: Aggregates multiple hypervectors into a similar result
- Permutation: Reorders elements to encode sequences
- Similarity: Measures relatedness between hypervectors
"""

from typing import Optional, Union, Callable
import jax
import jax.numpy as jnp


# ============================================================================
# Binary Spatter Code (BSC) Operations
# ============================================================================


@jax.jit
def bind_bsc(x: jax.Array, y: jax.Array) -> jax.Array:
    """Bind two hypervectors using XOR for Binary Spatter Codes.

    Binding creates a new hypervector that is dissimilar to both inputs.
    For BSC, binding is implemented as element-wise XOR.

    Args:
        x: Binary hypervector of shape (..., d)
        y: Binary hypervector of shape (..., d)

    Returns:
        Bound hypervector of shape (..., d), dissimilar to both x and y

    Properties:
        - Commutative: bind(x, y) = bind(y, x)
        - Self-inverse: bind(bind(x, y), y) = x
        - Produces dissimilar result: similarity(bind(x, y), x) ≈ 0.5

    Example:
        >>> import jax.numpy as jnp
        >>> x = jnp.array([True, False, True, False])
        >>> y = jnp.array([True, True, False, False])
        >>> bind_bsc(x, y)
        Array([False, True, True, False], dtype=bool)
    """
    return jnp.logical_xor(x, y)


def bundle_bsc(vectors: jax.Array, axis: int = 0) -> jax.Array:
    """Bundle hypervectors using majority rule for Binary Spatter Codes.

    Bundling creates a new hypervector similar to all inputs by taking
    the majority vote at each dimension.

    Args:
        vectors: Binary hypervectors of shape with axis containing vectors to bundle
        axis: Axis along which to bundle (default: 0)

    Returns:
        Bundled hypervector, similar to all inputs

    Properties:
        - Commutative: order doesn't matter
        - Associative: can bundle in groups
        - Result is similar to all inputs

    Example:
        >>> vectors = jnp.array([[True, False, True],
        ...                      [True, True, False],
        ...                      [False, True, True]])
        >>> bundle_bsc(vectors, axis=0)
        Array([True, True, True], dtype=bool)  # Majority at each position
    """
    counts = jnp.sum(vectors, axis=axis)
    # Get shape along axis - use static axis value
    shape_size = vectors.shape[axis]
    threshold = shape_size / 2.0
    return counts > threshold


@jax.jit
def inverse_bsc(x: jax.Array) -> jax.Array:
    """Compute inverse for BSC (identity since XOR is self-inverse).

    For Binary Spatter Codes, the inverse is the vector itself because
    XOR is self-inverse: x XOR y XOR y = x.

    Args:
        x: Binary hypervector of shape (..., d)

    Returns:
        Same hypervector (identity operation)

    Example:
        >>> x = jnp.array([True, False, True, False])
        >>> inverse_bsc(x)
        Array([True, False, True, False], dtype=bool)
    """
    return x  # XOR is self-inverse


@jax.jit
def hamming_similarity(x: jax.Array, y: jax.Array) -> jax.Array:
    """Compute normalized Hamming similarity between binary hypervectors.

    Returns the fraction of matching bits between two binary vectors.
    Random vectors have similarity ≈ 0.5.

    Args:
        x: Binary hypervector of shape (..., d)
        y: Binary hypervector of shape (..., d)

    Returns:
        Similarity score in [0, 1], where 1 is identical and 0.5 is random

    Example:
        >>> x = jnp.array([True, False, True, False])
        >>> y = jnp.array([True, True, True, False])
        >>> hamming_similarity(x, y)
        Array(0.75, dtype=float32)  # 3 out of 4 match
    """
    matches = jnp.logical_not(jnp.logical_xor(x, y))
    return jnp.mean(matches.astype(jnp.float32), axis=-1)


# ============================================================================
# Multiply-Add-Permute (MAP) Operations
# ============================================================================


@jax.jit
def bind_map(x: jax.Array, y: jax.Array) -> jax.Array:
    """Bind two hypervectors using element-wise multiplication for MAP.

    For real-valued vectors (MAP model), binding is element-wise multiplication.
    The result is dissimilar to both inputs.

    Args:
        x: Real-valued hypervector of shape (..., d)
        y: Real-valued hypervector of shape (..., d)

    Returns:
        Bound hypervector of shape (..., d)

    Properties:
        - Commutative: bind(x, y) = bind(y, x)
        - Inverse via element-wise reciprocal
        - Preserves unit norm if inputs are normalized

    Example:
        >>> x = jnp.array([1.0, -1.0, 0.5, -0.5])
        >>> y = jnp.array([0.5, 0.5, -1.0, 1.0])
        >>> bind_map(x, y)
        Array([0.5, -0.5, -0.5, -0.5], dtype=float32)
    """
    return x * y


def bundle_map(vectors: jax.Array, axis: int = 0) -> jax.Array:
    """Bundle hypervectors using normalized sum for MAP.

    For real-valued vectors, bundling is the normalized sum. The result
    is similar to all inputs (high cosine similarity).

    Args:
        vectors: Real-valued hypervectors with axis containing vectors to bundle
        axis: Axis along which to bundle (default: 0)

    Returns:
        Bundled and normalized hypervector

    Example:
        >>> vectors = jnp.array([[1.0, 0.0, 0.0],
        ...                      [0.0, 1.0, 0.0],
        ...                      [0.0, 0.0, 1.0]])
        >>> bundle_map(vectors, axis=0)
        Array([0.577, 0.577, 0.577], dtype=float32)  # Normalized average
    """
    summed = jnp.sum(vectors, axis=axis)
    norm = jnp.linalg.norm(summed, axis=-1, keepdims=True)
    return summed / (norm + 1e-8)


@jax.jit
def inverse_map(x: jax.Array, eps: float = 1e-8) -> jax.Array:
    """Compute inverse for MAP using element-wise reciprocal.

    For MAP binding (element-wise multiplication), the inverse is
    element-wise reciprocal: bind(bind(x, y), inverse(y)) = x.

    Args:
        x: Real-valued hypervector of shape (..., d)
        eps: Small constant to avoid division by zero (default: 1e-8)

    Returns:
        Inverse hypervector

    Example:
        >>> x = jnp.array([2.0, 0.5, -1.0, 4.0])
        >>> inverse_map(x)
        Array([0.5, 2.0, -1.0, 0.25], dtype=float32)
    """
    return 1.0 / (x + eps * jnp.sign(x))


@jax.jit
def cosine_similarity(x: jax.Array, y: jax.Array) -> jax.Array:
    """Compute cosine similarity between real-valued hypervectors.

    Returns the cosine of the angle between two vectors. Random unit vectors
    have similarity ≈ 0.

    Args:
        x: Real-valued hypervector of shape (..., d)
        y: Real-valued hypervector of shape (..., d)

    Returns:
        Similarity score in [-1, 1], where 1 is identical, -1 is opposite,
        and 0 is orthogonal

    Example:
        >>> x = jnp.array([1.0, 0.0, 0.0])
        >>> y = jnp.array([0.0, 1.0, 0.0])
        >>> cosine_similarity(x, y)
        Array(0., dtype=float32)  # Orthogonal vectors
    """
    x_norm = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
    y_norm = y / (jnp.linalg.norm(y, axis=-1, keepdims=True) + 1e-8)
    return jnp.sum(x_norm * y_norm, axis=-1)


# ============================================================================
# Universal Operations (work with all VSA models)
# ============================================================================


@jax.jit
def permute(x: jax.Array, shifts: int = 1) -> jax.Array:
    """Cyclically permute a hypervector to encode sequence information.

    Permutation reorders elements to represent positional or sequential
    information. Cyclic shifts preserve the distribution of values.

    Args:
        x: Hypervector of shape (..., d)
        shifts: Number of positions to shift (default: 1)

    Returns:
        Permuted hypervector of shape (..., d)

    Properties:
        - Invertible: permute(permute(x, k), -k) = x
        - Preserves similarity structure
        - Used for encoding sequences: [a, b, c] -> permute(a, 2) + permute(b, 1) + c

    Example:
        >>> x = jnp.array([1, 2, 3, 4, 5])
        >>> permute(x, shifts=2)
        Array([4, 5, 1, 2, 3], dtype=int32)
    """
    return jnp.roll(x, shifts, axis=-1)


@jax.jit
def cleanup(
    query: jax.Array,
    memory: jax.Array,
    similarity_fn: Callable[[jax.Array, jax.Array], jax.Array] = cosine_similarity,
    return_similarity: bool = False,
) -> Union[jax.Array, tuple[jax.Array, jax.Array]]:
    """Find the most similar vector in memory to the query.

    Cleanup (or resonator) is used to retrieve the closest known hypervector
    from memory, useful for error correction and symbol retrieval.

    Args:
        query: Query hypervector of shape (..., d)
        memory: Memory hypervectors of shape (n, d)
        similarity_fn: Function to compute similarity (default: cosine_similarity)
        return_similarity: Whether to return similarity scores (default: False)

    Returns:
        Most similar vector from memory, or (vector, similarity) if return_similarity=True

    Example:
        >>> query = jnp.array([0.9, 0.1, 0.0])
        >>> memory = jnp.array([[1.0, 0.0, 0.0],
        ...                     [0.0, 1.0, 0.0],
        ...                     [0.0, 0.0, 1.0]])
        >>> cleanup(query, memory)
        Array([1., 0., 0.], dtype=float32)  # Closest to first vector
    """
    # Compute similarities with all memory vectors
    similarities = jax.vmap(lambda m: similarity_fn(query, m))(memory)

    # Find most similar
    best_idx = jnp.argmax(similarities)
    best_vector = memory[best_idx]

    if return_similarity:
        return best_vector, similarities[best_idx]
    return best_vector


# ============================================================================
# Batch Operations (using vmap)
# ============================================================================

# Batch versions for common operations
batch_bind_bsc = jax.vmap(bind_bsc, in_axes=(0, 0))
batch_bind_map = jax.vmap(bind_map, in_axes=(0, 0))
batch_hamming_similarity = jax.vmap(hamming_similarity, in_axes=(0, None))
batch_cosine_similarity = jax.vmap(cosine_similarity, in_axes=(0, None))


# ============================================================================
# Holographic Reduced Representations (HRR) Operations
# ============================================================================


@jax.jit
def bind_hrr(x: jax.Array, y: jax.Array) -> jax.Array:
    """Bind two hypervectors using circular convolution for HRR.

    Circular convolution in the spatial domain is equivalent to element-wise
    multiplication in the Fourier domain, making it efficient to compute.

    Args:
        x: Real-valued hypervector of shape (..., d)
        y: Real-valued hypervector of shape (..., d)

    Returns:
        Bound hypervector via circular convolution

    Note:
        This is computed efficiently using FFT: ifft(fft(x) * fft(y))
    """
    # Compute via FFT for efficiency
    x_fft = jnp.fft.fft(x, axis=-1)
    y_fft = jnp.fft.fft(y, axis=-1)
    result_fft = x_fft * y_fft
    return jnp.fft.ifft(result_fft, axis=-1).real


@jax.jit
def inverse_hrr(x: jax.Array) -> jax.Array:
    """Compute inverse for HRR (reverse the circular convolution).

    For HRR, the inverse reverses the order of elements (except the first).

    Args:
        x: Real-valued hypervector of shape (..., d)

    Returns:
        Inverse hypervector
    """
    # Reverse all elements except the first
    return jnp.concatenate([x[..., :1], jnp.flip(x[..., 1:], axis=-1)], axis=-1)


# Bundle for HRR is same as MAP
bundle_hrr = bundle_map


__all__ = [
    # BSC operations
    "bind_bsc",
    "bundle_bsc",
    "inverse_bsc",
    "hamming_similarity",
    # MAP operations
    "bind_map",
    "bundle_map",
    "inverse_map",
    "cosine_similarity",
    # HRR operations
    "bind_hrr",
    "bundle_hrr",
    "inverse_hrr",
    # Universal operations
    "permute",
    "cleanup",
    # Batch operations
    "batch_bind_bsc",
    "batch_bind_map",
    "batch_hamming_similarity",
    "batch_cosine_similarity",
]
