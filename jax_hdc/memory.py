"""Memory modules for Hyperdimensional Computing.

Sparse Distributed Memory (SDM), Hopfield networks, and attention-based retrieval.
"""

from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from typing import Optional

import jax
import jax.numpy as jnp

from jax_hdc import functional as F

try:
    _test_cls = type("_Test", (), {"__annotations__": {"x": int}})
    jax.tree_util.register_dataclass(_test_cls)
    del _test_cls

    def _register_dataclass(cls):
        return jax.tree_util.register_dataclass(cls)

except TypeError:
    import dataclasses as _dc

    def _register_dataclass(cls):
        data_f = [f.name for f in _dc.fields(cls) if not f.metadata.get("static", False)]
        meta_f = [f.name for f in _dc.fields(cls) if f.metadata.get("static", False)]
        return jax.tree_util.register_dataclass(cls, data_f, meta_f)


@_register_dataclass
@dataclass
class SparseDistributedMemory:
    """Sparse Distributed Memory (SDM) for content-addressable storage.

    Stores patterns in a fixed set of locations; retrieval finds locations
    within a radius of the query and sums their contents.
    """

    locations: jax.Array  # (num_locations, dimensions)
    contents: jax.Array  # (num_locations, dimensions)
    dimensions: int = field(metadata=dict(static=True))
    radius: float = field(metadata=dict(static=True))

    @staticmethod
    def create(
        num_locations: int,
        dimensions: int,
        radius: float = 0.0,
        key: Optional[jax.Array] = None,
    ) -> "SparseDistributedMemory":
        if key is None:
            key = jax.random.PRNGKey(0)
        locs = jax.random.normal(key, (num_locations, dimensions))
        locs = locs / (jnp.linalg.norm(locs, axis=-1, keepdims=True) + 1e-8)
        contents = jnp.zeros((num_locations, dimensions))
        return SparseDistributedMemory(
            locations=locs,
            contents=contents,
            dimensions=dimensions,
            radius=radius,
        )

    def write(self, address: jax.Array, value: jax.Array) -> "SparseDistributedMemory":
        """Write value to locations near address (cosine sim > threshold)."""
        sims = jax.vmap(lambda loc: F.cosine_similarity(address, loc))(self.locations)
        mask = sims >= (1.0 - self.radius)
        delta = mask[:, None].astype(jnp.float32) * value
        return SparseDistributedMemory(
            locations=self.locations,
            contents=self.contents + delta,
            dimensions=self.dimensions,
            radius=self.radius,
        )

    @jax.jit
    def read(self, address: jax.Array) -> jax.Array:
        """Read from locations near address."""
        sims = jax.vmap(lambda loc: F.cosine_similarity(address, loc))(self.locations)
        mask = sims >= (1.0 - self.radius)
        summed = jnp.sum(self.contents * mask[:, None], axis=0)
        norm = jnp.linalg.norm(summed) + 1e-8
        return summed / norm


@_register_dataclass
@dataclass
class HopfieldMemory:
    """Modern Hopfield network for associative memory."""

    patterns: jax.Array  # (num_patterns, dimensions)
    dimensions: int = field(metadata=dict(static=True))
    beta: float = field(metadata=dict(static=True), default=1.0)

    @staticmethod
    def create(
        dimensions: int,
        beta: float = 1.0,
    ) -> "HopfieldMemory":
        return HopfieldMemory(
            patterns=jnp.zeros((0, dimensions)),
            dimensions=dimensions,
            beta=beta,
        )

    def add(self, pattern: jax.Array) -> "HopfieldMemory":
        p = pattern.reshape(-1) / (jnp.linalg.norm(pattern) + 1e-8)
        new_patterns = jnp.concatenate([self.patterns, p[None, :]], axis=0)
        return dc_replace(self, patterns=new_patterns)

    @jax.jit
    def retrieve(self, query: jax.Array) -> jax.Array:
        """Retrieve most similar pattern via softmax attention."""
        q = query.reshape(-1)
        if self.patterns.shape[0] == 0:
            return jnp.zeros_like(q)
        sims = jax.vmap(lambda p: F.cosine_similarity(q, p))(self.patterns)
        weights = jax.nn.softmax(self.beta * sims)
        return jnp.sum(self.patterns * weights[:, None], axis=0)


__all__ = ["SparseDistributedMemory", "HopfieldMemory"]
