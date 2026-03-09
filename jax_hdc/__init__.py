"""JAX-HDC: Hyperdimensional Computing with JAX."""

__version__ = "0.1.0-alpha"

from jax_hdc import embeddings, functional, memory, models, utils, vsa
from jax_hdc.embeddings import (
    GraphEncoder,
    KernelEncoder,
    LevelEncoder,
    ProjectionEncoder,
    RandomEncoder,
)
from jax_hdc.functional import (
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
from jax_hdc.memory import AttentionMemory, HopfieldMemory, SparseDistributedMemory
from jax_hdc.models import AdaptiveHDC, CentroidClassifier, LVQClassifier, RegularizedLSClassifier
from jax_hdc.vsa import BSBC, BSC, CGR, FHRR, HRR, MAP, MCR, VTB

__all__ = [
    "functional",
    "vsa",
    "embeddings",
    "models",
    "utils",
    "memory",
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
    "BSC",
    "BSBC",
    "MAP",
    "HRR",
    "FHRR",
    "CGR",
    "MCR",
    "VTB",
    "RandomEncoder",
    "LevelEncoder",
    "ProjectionEncoder",
    "KernelEncoder",
    "GraphEncoder",
    "CentroidClassifier",
    "AdaptiveHDC",
    "LVQClassifier",
    "RegularizedLSClassifier",
    "SparseDistributedMemory",
    "HopfieldMemory",
    "AttentionMemory",
]
