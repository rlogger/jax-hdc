"""JAX-HDC: A high-performance JAX library for Hyperdimensional Computing.

This library provides efficient implementations of Hyperdimensional Computing (HDC)
and Vector Symbolic Architectures (VSA) using JAX for hardware acceleration.
"""

__version__ = "0.1.0-alpha"

from jax_hdc import embeddings, functional, models, utils, vsa

# Embeddings
from jax_hdc.embeddings import LevelEncoder, ProjectionEncoder, RandomEncoder

# Core functional operations
from jax_hdc.functional import (  # BSC operations; MAP operations; Universal operations
    bind_bsc,
    bind_map,
    bundle_bsc,
    bundle_map,
    cleanup,
    cosine_similarity,
    hamming_similarity,
    inverse_bsc,
    inverse_map,
    permute,
)

# Models
from jax_hdc.models import AdaptiveHDC, CentroidClassifier

# VSA models
from jax_hdc.vsa import BSC, FHRR, HRR, MAP

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
    "ProjectionEncoder",
    # Models
    "CentroidClassifier",
    "AdaptiveHDC",
]
