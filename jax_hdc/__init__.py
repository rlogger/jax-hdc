"""JAX-HDC: Hyperdimensional Computing with JAX."""

__version__ = "0.1.0-alpha"

from jax_hdc import embeddings, functional, memory, models, structures, utils, vsa
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
    bind_sequence,
    bundle_bsc,
    bundle_map,
    bundle_sequence,
    cleanup,
    cosine_similarity,
    cross_product,
    dot_similarity,
    graph_encode,
    hamming_similarity,
    hash_table,
    inverse_bsc,
    inverse_map,
    multibind_bsc,
    multibind_map,
    negative_bsc,
    negative_map,
    ngrams,
    permute,
    resonator,
)
from jax_hdc.memory import AttentionMemory, HopfieldMemory, SparseDistributedMemory
from jax_hdc.models import AdaptiveHDC, CentroidClassifier, LVQClassifier, RegularizedLSClassifier
from jax_hdc.structures import Graph, HashTable, Multiset, Sequence
from jax_hdc.vsa import BSBC, BSC, CGR, FHRR, HRR, MAP, MCR, VTB

__all__ = [
    # Modules
    "functional",
    "vsa",
    "embeddings",
    "models",
    "utils",
    "memory",
    "structures",
    # Core operations
    "bind_bsc",
    "bundle_bsc",
    "inverse_bsc",
    "negative_bsc",
    "hamming_similarity",
    "bind_map",
    "bundle_map",
    "inverse_map",
    "negative_map",
    "cosine_similarity",
    "dot_similarity",
    "permute",
    "cleanup",
    # Multi-vector operations
    "multibind_map",
    "multibind_bsc",
    "cross_product",
    # Composite encodings
    "hash_table",
    "ngrams",
    "bundle_sequence",
    "bind_sequence",
    "graph_encode",
    "resonator",
    # VSA models
    "BSC",
    "BSBC",
    "MAP",
    "HRR",
    "FHRR",
    "CGR",
    "MCR",
    "VTB",
    # Encoders
    "RandomEncoder",
    "LevelEncoder",
    "ProjectionEncoder",
    "KernelEncoder",
    "GraphEncoder",
    # Classifiers
    "CentroidClassifier",
    "AdaptiveHDC",
    "LVQClassifier",
    "RegularizedLSClassifier",
    # Memory
    "SparseDistributedMemory",
    "HopfieldMemory",
    "AttentionMemory",
    # Structures
    "Multiset",
    "HashTable",
    "Sequence",
    "Graph",
]
