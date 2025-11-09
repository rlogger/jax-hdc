"""JAX-HDC: A high-performance JAX library for Hyperdimensional Computing.

This library provides efficient implementations of Hyperdimensional Computing (HDC)
and Vector Symbolic Architectures (VSA) using JAX for hardware acceleration.
"""

__version__ = "0.1.0-alpha"

from jax_hdc import functional
from jax_hdc import vsa
from jax_hdc import embeddings
from jax_hdc import models
from jax_hdc import utils

# Core functional operations
from jax_hdc.functional import (
    # BSC operations
    bind_bsc,
    bundle_bsc,
    inverse_bsc,
    hamming_similarity,
    # MAP operations
    bind_map,
    bundle_map,
    inverse_map,
    cosine_similarity,
    # Universal operations
    permute,
    cleanup,
)

# VSA models
from jax_hdc.vsa import BSC, MAP, HRR, FHRR

# Embeddings
from jax_hdc.embeddings import RandomEncoder, LevelEncoder

# Models
from jax_hdc.models import CentroidClassifier

__all__ = [
    # Modules
    "functional",
    "vsa",
    "embeddings",
    "models",
    "utils",
    # Functional operations
    "bind_bsc",
    "bundle_bsc",
    "inverse_bsc",
    "hamming_similarity",
    "bind_map",
    "bundle_map",
    "inverse_map",
    "cosine_similarity",
    "permute",
    "cleanup",
    # VSA models
    "BSC",
    "MAP",
    "HRR",
    "FHRR",
    # Embeddings
    "RandomEncoder",
    "LevelEncoder",
    # Models
    "CentroidClassifier",
]
