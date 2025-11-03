"""Vector Symbolic Architecture (VSA) model implementations.

This module provides different VSA models, each with their own binding,
bundling, and similarity operations. All models follow a consistent API.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable
import jax
import jax.numpy as jnp
from jax_hdc import functional as F


@jax.tree_util.register_dataclass
@dataclass
class VSAModel:
    """Base class for VSA models defining the interface."""

    name: str = field(metadata=dict(static=True))
    dimensions: int = field(metadata=dict(static=True))

    def bind(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Bind two hypervectors."""
        raise NotImplementedError

    def bundle(self, vectors: jax.Array, axis: int = 0) -> jax.Array:
        """Bundle multiple hypervectors."""
        raise NotImplementedError

    def inverse(self, x: jax.Array) -> jax.Array:
        """Compute the inverse of a hypervector."""
        raise NotImplementedError

    def similarity(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Compute similarity between hypervectors."""
        raise NotImplementedError

    def random(self, key: jax.Array, shape: tuple) -> jax.Array:
        """Generate random hypervectors."""
        raise NotImplementedError


@jax.tree_util.register_dataclass
@dataclass
class BSC(VSAModel):
    """Binary Spatter Codes (BSC) model.

    BSC uses binary hypervectors with:
    - Binding: XOR (element-wise logical XOR)
    - Bundling: Majority (element-wise majority vote)
    - Similarity: Hamming distance

    Properties:
        - Memory efficient (1 bit per dimension)
        - Fast operations (bitwise operations)
        - Self-inverse binding

    Example:
        >>> import jax
        >>> model = BSC.create(dimensions=10000)
        >>> key = jax.random.PRNGKey(42)
        >>> x = model.random(key, (10000,))
        >>> y = model.random(key, (10000,))
        >>> bound = model.bind(x, y)
        >>> sim = model.similarity(bound, x)
        >>> print(f"Similarity: {sim:.3f}")  # Should be ~0.5 (random)
    """

    name: str = field(default="bsc", metadata=dict(static=True))

    @staticmethod
    def create(dimensions: int = 10000) -> "BSC":
        """Create a BSC model.

        Args:
            dimensions: Dimensionality of hypervectors (default: 10000)

        Returns:
            Initialized BSC model
        """
        return BSC(name="bsc", dimensions=dimensions)

    @jax.jit
    def bind(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Bind using XOR."""
        return F.bind_bsc(x, y)

    @jax.jit
    def bundle(self, vectors: jax.Array, axis: int = 0) -> jax.Array:
        """Bundle using majority rule."""
        return F.bundle_bsc(vectors, axis=axis)

    @jax.jit
    def inverse(self, x: jax.Array) -> jax.Array:
        """Inverse is identity for XOR."""
        return F.inverse_bsc(x)

    @jax.jit
    def similarity(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Compute Hamming similarity."""
        return F.hamming_similarity(x, y)

    def random(self, key: jax.Array, shape: tuple) -> jax.Array:
        """Generate random binary hypervectors.

        Args:
            key: JAX random key
            shape: Shape of output array

        Returns:
            Random binary hypervectors with ~50% ones
        """
        return jax.random.bernoulli(key, 0.5, shape=shape)


@jax.tree_util.register_dataclass
@dataclass
class MAP(VSAModel):
    """Multiply-Add-Permute (MAP) model.

    MAP uses real-valued hypervectors with:
    - Binding: Element-wise multiplication
    - Bundling: Normalized sum
    - Similarity: Cosine similarity

    Properties:
        - Smooth optimization landscape
        - Compatible with gradient-based learning
        - Simple inverse operation

    Example:
        >>> import jax
        >>> model = MAP.create(dimensions=10000)
        >>> key = jax.random.PRNGKey(42)
        >>> x = model.random(key, (10000,))
        >>> y = model.random(key, (10000,))
        >>> bound = model.bind(x, y)
        >>> sim = model.similarity(bound, x)
        >>> print(f"Similarity: {sim:.3f}")  # Should be ~0.0 (orthogonal)
    """

    name: str = field(default="map", metadata=dict(static=True))

    @staticmethod
    def create(dimensions: int = 10000) -> "MAP":
        """Create a MAP model.

        Args:
            dimensions: Dimensionality of hypervectors (default: 10000)

        Returns:
            Initialized MAP model
        """
        return MAP(name="map", dimensions=dimensions)

    @jax.jit
    def bind(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Bind using element-wise multiplication."""
        return F.bind_map(x, y)

    @jax.jit
    def bundle(self, vectors: jax.Array, axis: int = 0) -> jax.Array:
        """Bundle using normalized sum."""
        return F.bundle_map(vectors, axis=axis)

    @jax.jit
    def inverse(self, x: jax.Array) -> jax.Array:
        """Inverse via element-wise reciprocal."""
        return F.inverse_map(x)

    @jax.jit
    def similarity(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Compute cosine similarity."""
        return F.cosine_similarity(x, y)

    def random(self, key: jax.Array, shape: tuple) -> jax.Array:
        """Generate random real-valued hypervectors.

        Args:
            key: JAX random key
            shape: Shape of output array

        Returns:
            Random normalized hypervectors sampled from normal distribution
        """
        vectors = jax.random.normal(key, shape=shape)
        # Normalize to unit length
        norm = jnp.linalg.norm(vectors, axis=-1, keepdims=True)
        return vectors / (norm + 1e-8)


@jax.tree_util.register_dataclass
@dataclass
class HRR(VSAModel):
    """Holographic Reduced Representations (HRR) model.

    HRR uses real-valued hypervectors with:
    - Binding: Circular convolution
    - Bundling: Normalized sum
    - Similarity: Cosine similarity

    Properties:
        - Theoretically well-founded (based on holography)
        - Efficient via FFT
        - Good compression properties

    Example:
        >>> import jax
        >>> model = HRR.create(dimensions=10000)
        >>> key = jax.random.PRNGKey(42)
        >>> x = model.random(key, (10000,))
        >>> y = model.random(key, (10000,))
        >>> bound = model.bind(x, y)
        >>> # Unbinding: bind(bound, inverse(y)) ≈ x
        >>> unbound = model.bind(bound, model.inverse(y))
        >>> sim = model.similarity(unbound, x)
        >>> print(f"Similarity: {sim:.3f}")  # Should be high
    """

    name: str = field(default="hrr", metadata=dict(static=True))

    @staticmethod
    def create(dimensions: int = 10000) -> "HRR":
        """Create an HRR model.

        Args:
            dimensions: Dimensionality of hypervectors (default: 10000)

        Returns:
            Initialized HRR model
        """
        return HRR(name="hrr", dimensions=dimensions)

    @jax.jit
    def bind(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Bind using circular convolution."""
        return F.bind_hrr(x, y)

    @jax.jit
    def bundle(self, vectors: jax.Array, axis: int = 0) -> jax.Array:
        """Bundle using normalized sum."""
        return F.bundle_hrr(vectors, axis=axis)

    @jax.jit
    def inverse(self, x: jax.Array) -> jax.Array:
        """Inverse via element reversal."""
        return F.inverse_hrr(x)

    @jax.jit
    def similarity(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Compute cosine similarity."""
        return F.cosine_similarity(x, y)

    def random(self, key: jax.Array, shape: tuple) -> jax.Array:
        """Generate random real-valued hypervectors.

        Args:
            key: JAX random key
            shape: Shape of output array

        Returns:
            Random normalized hypervectors sampled from normal distribution
        """
        vectors = jax.random.normal(key, shape=shape)
        # Normalize to unit length
        norm = jnp.linalg.norm(vectors, axis=-1, keepdims=True)
        return vectors / (norm + 1e-8)


@jax.tree_util.register_dataclass
@dataclass
class FHRR(VSAModel):
    """Fourier Holographic Reduced Representations (FHRR) model.

    FHRR uses complex-valued hypervectors with:
    - Binding: Element-wise multiplication in Fourier domain
    - Bundling: Normalized sum
    - Similarity: Cosine similarity (magnitude)

    Properties:
        - Efficient binding (no FFT needed)
        - Enables vector function architectures
        - Phase information for structured representations

    Example:
        >>> import jax
        >>> model = FHRR.create(dimensions=10000)
        >>> key = jax.random.PRNGKey(42)
        >>> x = model.random(key, (10000,))
        >>> y = model.random(key, (10000,))
        >>> bound = model.bind(x, y)
    """

    name: str = field(default="fhrr", metadata=dict(static=True))

    @staticmethod
    def create(dimensions: int = 10000) -> "FHRR":
        """Create an FHRR model.

        Args:
            dimensions: Dimensionality of hypervectors (default: 10000)

        Returns:
            Initialized FHRR model
        """
        return FHRR(name="fhrr", dimensions=dimensions)

    @jax.jit
    def bind(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Bind using element-wise multiplication."""
        return x * y

    @jax.jit
    def bundle(self, vectors: jax.Array, axis: int = 0) -> jax.Array:
        """Bundle using normalized sum."""
        summed = jnp.sum(vectors, axis=axis)
        norm = jnp.linalg.norm(summed, axis=-1, keepdims=True)
        return summed / (norm + 1e-8)

    @jax.jit
    def inverse(self, x: jax.Array) -> jax.Array:
        """Inverse via complex conjugate."""
        return jnp.conj(x)

    @jax.jit
    def similarity(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Compute cosine similarity of complex vectors."""
        x_norm = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
        y_norm = y / (jnp.linalg.norm(y, axis=-1, keepdims=True) + 1e-8)
        # Use real part of inner product
        return jnp.real(jnp.sum(x_norm * jnp.conj(y_norm), axis=-1))

    def random(self, key: jax.Array, shape: tuple) -> jax.Array:
        """Generate random complex hypervectors on unit circle.

        Args:
            key: JAX random key
            shape: Shape of output array

        Returns:
            Random unit complex hypervectors
        """
        # Random phases on unit circle
        phases = jax.random.uniform(key, shape=shape, minval=0, maxval=2*jnp.pi)
        return jnp.exp(1j * phases)


def create_vsa_model(model_type: str = "map", dimensions: int = 10000) -> VSAModel:
    """Factory function to create VSA models.

    Args:
        model_type: Type of VSA model ('bsc', 'map', 'hrr', 'fhrr')
        dimensions: Dimensionality of hypervectors (default: 10000)

    Returns:
        Initialized VSA model

    Example:
        >>> model = create_vsa_model('map', dimensions=10000)
        >>> print(model.name)
        'map'
    """
    models = {
        "bsc": BSC,
        "map": MAP,
        "hrr": HRR,
        "fhrr": FHRR,
    }

    if model_type not in models:
        raise ValueError(
            f"Unknown VSA model: {model_type}. "
            f"Available models: {list(models.keys())}"
        )

    return models[model_type].create(dimensions=dimensions)


__all__ = [
    "VSAModel",
    "BSC",
    "MAP",
    "HRR",
    "FHRR",
    "create_vsa_model",
]
