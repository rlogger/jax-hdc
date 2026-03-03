"""Tests for memory modules."""

import jax
import jax.numpy as jnp
import pytest

from jax_hdc.memory import HopfieldMemory, SparseDistributedMemory


class TestSparseDistributedMemory:
    """Tests for SparseDistributedMemory."""

    def test_create_with_key(self):
        """Test SDM creation with explicit key."""
        key = jax.random.PRNGKey(42)
        sdm = SparseDistributedMemory.create(
            num_locations=50, dimensions=100, radius=0.2, key=key
        )
        assert sdm.locations.shape == (50, 100)
        assert sdm.contents.shape == (50, 100)
        assert sdm.dimensions == 100
        assert sdm.radius == 0.2

    def test_create_without_key(self):
        """Test SDM creation with default key."""
        sdm = SparseDistributedMemory.create(num_locations=20, dimensions=50)
        assert sdm.locations.shape == (20, 50)

    def test_write_and_read(self):
        """Test write and read operations."""
        key = jax.random.PRNGKey(42)
        sdm = SparseDistributedMemory.create(
            num_locations=100, dimensions=100, radius=0.3, key=key
        )

        addr = jax.random.normal(key, (100,))
        addr = addr / (jnp.linalg.norm(addr) + 1e-8)
        val = jax.random.normal(jax.random.split(key)[1], (100,))

        sdm = sdm.write(addr, val)
        result = sdm.read(addr)

        assert result.shape == (100,)
        assert jnp.isfinite(result).all()

    def test_read_normalized_output(self):
        """Test that read returns normalized vector."""
        key = jax.random.PRNGKey(42)
        sdm = SparseDistributedMemory.create(
            num_locations=50, dimensions=50, radius=0.5, key=key
        )
        addr = sdm.locations[0]
        val = jax.random.normal(key, (50,))
        sdm = sdm.write(addr, val)
        result = sdm.read(addr)
        norm = jnp.linalg.norm(result)
        assert jnp.allclose(norm, 1.0, atol=1e-5)


class TestHopfieldMemory:
    """Tests for HopfieldMemory."""

    def test_create_default_beta(self):
        """Test Hopfield creation with default beta."""
        hop = HopfieldMemory.create(dimensions=100)
        assert hop.dimensions == 100
        assert hop.beta == 1.0
        assert hop.patterns.shape == (0, 100)

    def test_create_custom_beta(self):
        """Test Hopfield creation with custom beta."""
        hop = HopfieldMemory.create(dimensions=100, beta=2.0)
        assert hop.beta == 2.0

    def test_add_and_retrieve(self):
        """Test adding patterns and retrieval."""
        hop = HopfieldMemory.create(dimensions=100)
        p = jax.random.normal(jax.random.PRNGKey(42), (100,))
        hop = hop.add(p)

        noisy = p + 0.1 * jax.random.normal(jax.random.PRNGKey(0), (100,))
        result = hop.retrieve(noisy)

        assert result.shape == (100,)
        assert jnp.isfinite(result).all()

    def test_retrieve_empty_memory(self):
        """Test retrieve on empty memory returns zeros."""
        hop = HopfieldMemory.create(dimensions=100)
        query = jax.random.normal(jax.random.PRNGKey(42), (100,))
        result = hop.retrieve(query)
        assert result.shape == (100,)
        assert jnp.allclose(result, 0.0)

    def test_add_multiple_patterns(self):
        """Test adding multiple patterns."""
        hop = HopfieldMemory.create(dimensions=50)
        for i in range(3):
            p = jax.random.normal(jax.random.PRNGKey(i), (50,))
            hop = hop.add(p)
        assert hop.patterns.shape == (3, 50)
