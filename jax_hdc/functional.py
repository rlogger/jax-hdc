"""Core functional operations for Hyperdimensional Computing.

This module provides the fundamental operations for manipulating hypervectors:
- Binding: Combines two hypervectors into a dissimilar result
- Bundling: Aggregates multiple hypervectors into a similar result
- Permutation: Reorders elements to encode sequences
- Similarity: Measures relatedness between hypervectors
"""

import functools
from typing import Callable, Union

import jax
import jax.numpy as jnp

from jax_hdc.constants import EPS


@jax.jit
def bind_bsc(x: jax.Array, y: jax.Array) -> jax.Array:
    """Bind two hypervectors using XOR for Binary Spatter Codes.

    Binding creates a new hypervector that is dissimilar to both inputs.

    Args:
        x: Binary hypervector of shape (..., d)
        y: Binary hypervector of shape (..., d)

    Returns:
        Bound hypervector of shape (..., d), dissimilar to both x and y
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
    """
    counts = jnp.sum(vectors, axis=axis)
    shape_size = vectors.shape[axis]
    threshold = shape_size / 2.0
    return counts > threshold


@jax.jit
def inverse_bsc(x: jax.Array) -> jax.Array:
    """Compute inverse for BSC (identity since XOR is self-inverse)."""
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
    """
    matches = jnp.logical_not(jnp.logical_xor(x, y))
    return jnp.mean(matches.astype(jnp.float32), axis=-1)


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
    """
    summed = jnp.sum(vectors, axis=axis)
    norm = jnp.linalg.norm(summed, axis=-1, keepdims=True)
    return summed / (norm + EPS)


@jax.jit
def inverse_map(x: jax.Array, eps: float = EPS) -> jax.Array:
    """Compute inverse for MAP using element-wise reciprocal.

    For MAP binding (element-wise multiplication), the inverse is
    element-wise reciprocal: bind(bind(x, y), inverse(y)) = x.
    Near-zero elements return 0 (no inverse; bind with 0 destroys information).

    Args:
        x: Real-valued hypervector of shape (..., d)
        eps: Small constant for numerical stability (default: EPS)

    Returns:
        Inverse hypervector
    """
    safe_inv = jnp.where(jnp.abs(x) > eps, 1.0 / x, 0.0)
    return safe_inv


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
    """
    x_norm = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + EPS)
    y_norm = y / (jnp.linalg.norm(y, axis=-1, keepdims=True) + EPS)
    return jnp.clip(jnp.sum(x_norm * y_norm, axis=-1), -1.0, 1.0)


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
    """
    return jnp.roll(x, shifts, axis=-1)


@functools.partial(jax.jit, static_argnames=("return_similarity",))
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
    """
    similarities = jax.vmap(lambda m: similarity_fn(query, m))(memory)
    best_idx = jnp.argmax(similarities)
    best_vector = memory[best_idx]

    if return_similarity:
        return best_vector, similarities[best_idx]
    return best_vector


# Batch versions for common operations
batch_bind_bsc = jax.vmap(bind_bsc, in_axes=(0, 0))
batch_bind_map = jax.vmap(bind_map, in_axes=(0, 0))
batch_hamming_similarity = jax.vmap(hamming_similarity, in_axes=(0, None))
batch_cosine_similarity = jax.vmap(cosine_similarity, in_axes=(0, None))


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
    """
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
    return jnp.concatenate([x[..., :1], jnp.flip(x[..., 1:], axis=-1)], axis=-1)


# HRR bundle reuses MAP bundle
bundle_hrr = bundle_map


@jax.jit
def bind_cgr(x: jax.Array, y: jax.Array, q: int) -> jax.Array:
    """Bind using modular addition for Cyclic Group Representation.

    Args:
        x: Integer hypervector with values in {0, ..., q-1}, shape (..., d)
        y: Integer hypervector with values in {0, ..., q-1}, shape (..., d)
        q: Size of the cyclic group

    Returns:
        Bound hypervector: (x + y) mod q
    """
    return (x + y) % q


def bundle_cgr(vectors: jax.Array, q: int, axis: int = 0) -> jax.Array:
    """Bundle using component-wise mode for CGR.

    Selects the most frequent value at each dimension.

    Args:
        vectors: Integer hypervectors with values in {0, ..., q-1}
        q: Size of the cyclic group
        axis: Axis along which to bundle (default: 0)

    Returns:
        Bundled hypervector with mode value at each dimension
    """
    one_hot = jax.nn.one_hot(vectors, q)
    counts = jnp.sum(one_hot, axis=axis)
    return jnp.argmax(counts, axis=-1).astype(jnp.int32)


@jax.jit
def inverse_cgr(x: jax.Array, q: int) -> jax.Array:
    """Inverse via modular negation for CGR.

    Args:
        x: Integer hypervector with values in {0, ..., q-1}
        q: Size of the cyclic group

    Returns:
        Inverse: (q - x) mod q
    """
    return (q - x) % q


@jax.jit
def matching_similarity(x: jax.Array, y: jax.Array) -> jax.Array:
    """Fraction of matching elements between integer hypervectors.

    Random vectors with q levels have expected similarity 1/q.

    Args:
        x: Integer hypervector of shape (..., d)
        y: Integer hypervector of shape (..., d)

    Returns:
        Similarity in [0, 1]
    """
    return jnp.mean((x == y).astype(jnp.float32), axis=-1)


# MCR bind reuses CGR bind
bind_mcr = bind_cgr
# MCR inverse reuses CGR inverse
inverse_mcr = inverse_cgr


def bundle_mcr(vectors: jax.Array, q: int, axis: int = 0) -> jax.Array:
    """Bundle using phasor sum and snap-to-grid for MCR.

    Converts indices to complex phasors (q-th roots of unity), sums them,
    and snaps back to the nearest discrete phase level.

    Args:
        vectors: Integer hypervectors with values in {0, ..., q-1}
        q: Number of phase levels
        axis: Axis along which to bundle (default: 0)

    Returns:
        Bundled hypervector with values snapped to nearest phase index
    """
    phases = 2 * jnp.pi * vectors.astype(jnp.float32) / q
    phasors = jnp.exp(1j * phases)
    summed = jnp.sum(phasors, axis=axis)
    result_angles = jnp.angle(summed) % (2 * jnp.pi)
    result_indices = jnp.round(result_angles * q / (2 * jnp.pi)) % q
    return result_indices.astype(jnp.int32)


@jax.jit
def phasor_similarity(x: jax.Array, y: jax.Array, q: int) -> jax.Array:
    """Similarity via phasor inner product for MCR.

    Converts to phasors and computes the real part of the normalized
    inner product. Random vectors have expected similarity ~0.

    Args:
        x: Integer hypervector with values in {0, ..., q-1}
        y: Integer hypervector with values in {0, ..., q-1}
        q: Number of phase levels

    Returns:
        Similarity in [-1, 1]
    """
    phases_x = 2 * jnp.pi * x.astype(jnp.float32) / q
    phases_y = 2 * jnp.pi * y.astype(jnp.float32) / q
    phasors_x = jnp.exp(1j * phases_x)
    phasors_y = jnp.exp(1j * phases_y)
    return jnp.real(jnp.mean(phasors_x * jnp.conj(phasors_y), axis=-1))


@jax.jit
def bind_vtb(x: jax.Array, y: jax.Array) -> jax.Array:
    """Bind using matrix multiplication for VTB.

    Reshapes d-dimensional vectors into sqrt(d) x sqrt(d) matrices
    and multiplies them. Requires d to be a perfect square.

    Args:
        x: Real-valued hypervector of shape (..., d)
        y: Real-valued hypervector of shape (..., d)

    Returns:
        Bound hypervector of shape (..., d)
    """
    d = x.shape[-1]
    n = round(d**0.5)
    X = x.reshape(*x.shape[:-1], n, n)
    Y = y.reshape(*y.shape[:-1], n, n)
    return (X @ Y).reshape(*x.shape[:-1], d)


@jax.jit
def inverse_vtb(x: jax.Array) -> jax.Array:
    """Inverse via matrix pseudoinverse for VTB."""
    d = x.shape[-1]
    n = round(d**0.5)
    X = x.reshape(*x.shape[:-1], n, n)
    return jnp.linalg.pinv(X).reshape(*x.shape[:-1], d)


# VTB bundle reuses MAP bundle
bundle_vtb = bundle_map


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
    # CGR operations
    "bind_cgr",
    "bundle_cgr",
    "inverse_cgr",
    "matching_similarity",
    # MCR operations
    "bind_mcr",
    "bundle_mcr",
    "inverse_mcr",
    "phasor_similarity",
    # VTB operations
    "bind_vtb",
    "bundle_vtb",
    "inverse_vtb",
    # Universal operations
    "permute",
    "cleanup",
    # Batch operations
    "batch_bind_bsc",
    "batch_bind_map",
    "batch_hamming_similarity",
    "batch_cosine_similarity",
]
