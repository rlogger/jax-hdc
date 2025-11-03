"""Tests for functional operations."""

import pytest
import jax
import jax.numpy as jnp
from jax_hdc import functional as F


class TestBSCOperations:
    """Test Binary Spatter Code operations."""

    def test_bind_bsc_commutativity(self):
        """Test that BSC binding is commutative."""
        key = jax.random.PRNGKey(42)
        x = jax.random.bernoulli(key, 0.5, shape=(10000,))
        y = jax.random.bernoulli(key, 0.5, shape=(10000,))

        result1 = F.bind_bsc(x, y)
        result2 = F.bind_bsc(y, x)

        assert jnp.allclose(result1, result2)

    def test_bind_bsc_self_inverse(self):
        """Test that BSC binding is self-inverse."""
        key = jax.random.PRNGKey(42)
        x = jax.random.bernoulli(key, 0.5, shape=(10000,))
        y = jax.random.bernoulli(key, 0.5, shape=(10000,))

        # bind(bind(x, y), y) should equal x
        bound = F.bind_bsc(x, y)
        unbound = F.bind_bsc(bound, y)

        assert jnp.allclose(unbound, x)

    def test_bind_bsc_dissimilarity(self):
        """Test that binding produces dissimilar vectors."""
        key = jax.random.PRNGKey(42)
        x = jax.random.bernoulli(key, 0.5, shape=(10000,))
        y = jax.random.bernoulli(key, 0.5, shape=(10000,))

        bound = F.bind_bsc(x, y)
        similarity = F.hamming_similarity(bound, x)

        # Should be around 0.5 for random vectors
        assert 0.4 < similarity < 0.6

    def test_bundle_bsc_similarity(self):
        """Test that bundling creates similar vector."""
        key = jax.random.PRNGKey(42)
        vectors = jax.random.bernoulli(key, 0.5, shape=(10, 10000))

        bundled = F.bundle_bsc(vectors, axis=0)

        # Bundled should be similar to each input
        for v in vectors:
            sim = F.hamming_similarity(bundled, v)
            assert sim > 0.5  # Should be noticeably similar

    def test_hamming_similarity_range(self):
        """Test that Hamming similarity is in [0, 1]."""
        key = jax.random.PRNGKey(42)
        x = jax.random.bernoulli(key, 0.5, shape=(10000,))
        y = jax.random.bernoulli(key, 0.5, shape=(10000,))

        sim = F.hamming_similarity(x, y)

        assert 0 <= sim <= 1

    def test_hamming_similarity_identity(self):
        """Test that identical vectors have similarity 1."""
        key = jax.random.PRNGKey(42)
        x = jax.random.bernoulli(key, 0.5, shape=(10000,))

        sim = F.hamming_similarity(x, x)

        assert jnp.allclose(sim, 1.0)


class TestMAPOperations:
    """Test Multiply-Add-Permute operations."""

    def test_bind_map_commutativity(self):
        """Test that MAP binding is commutative."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, shape=(10000,))
        y = jax.random.normal(key, shape=(10000,))

        result1 = F.bind_map(x, y)
        result2 = F.bind_map(y, x)

        assert jnp.allclose(result1, result2)

    def test_bind_map_inverse(self):
        """Test MAP binding inverse operation."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, shape=(10000,))
        y = jax.random.normal(key, shape=(10000,))

        # bind(bind(x, y), inverse(y)) should be close to x
        bound = F.bind_map(x, y)
        y_inv = F.inverse_map(y)
        unbound = F.bind_map(bound, y_inv)

        # Normalize both for comparison
        x_norm = x / (jnp.linalg.norm(x) + 1e-8)
        unbound_norm = unbound / (jnp.linalg.norm(unbound) + 1e-8)

        similarity = jnp.sum(x_norm * unbound_norm)
        assert similarity > 0.9  # Should be highly similar

    def test_bundle_map_normalization(self):
        """Test that MAP bundling produces normalized vectors."""
        key = jax.random.PRNGKey(42)
        vectors = jax.random.normal(key, shape=(10, 10000))

        bundled = F.bundle_map(vectors, axis=0)
        norm = jnp.linalg.norm(bundled)

        assert jnp.allclose(norm, 1.0, atol=1e-6)

    def test_cosine_similarity_range(self):
        """Test that cosine similarity is in [-1, 1]."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, shape=(10000,))
        y = jax.random.normal(key, shape=(10000,))

        sim = F.cosine_similarity(x, y)

        assert -1 <= sim <= 1

    def test_cosine_similarity_identity(self):
        """Test that identical vectors have similarity 1."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, shape=(10000,))

        sim = F.cosine_similarity(x, x)

        assert jnp.allclose(sim, 1.0, atol=1e-6)

    def test_cosine_similarity_orthogonal(self):
        """Test that orthogonal vectors have similarity ~0."""
        # Create orthogonal vectors
        x = jnp.zeros(10000)
        x = x.at[0].set(1.0)

        y = jnp.zeros(10000)
        y = y.at[1].set(1.0)

        sim = F.cosine_similarity(x, y)

        assert jnp.allclose(sim, 0.0, atol=1e-6)


class TestUniversalOperations:
    """Test operations that work with all VSA models."""

    def test_permute_invertibility(self):
        """Test that permutation is invertible."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, shape=(10000,))

        # Permute and unpermute
        permuted = F.permute(x, shifts=5)
        unpermuted = F.permute(permuted, shifts=-5)

        assert jnp.allclose(unpermuted, x)

    def test_permute_cyclic(self):
        """Test that permutation is cyclic."""
        x = jnp.arange(10)

        # Full cycle should return to original
        result = x
        for _ in range(10):
            result = F.permute(result, shifts=1)

        assert jnp.allclose(result, x)

    def test_cleanup_retrieval(self):
        """Test cleanup finds most similar vector."""
        key = jax.random.PRNGKey(42)

        # Create memory with 3 vectors
        memory = jax.random.normal(key, shape=(3, 10000))
        memory = memory / jnp.linalg.norm(memory, axis=-1, keepdims=True)

        # Query is a noisy version of first vector
        query = memory[0] + 0.1 * jax.random.normal(key, shape=(10000,))
        query = query / jnp.linalg.norm(query)

        # Cleanup should return first vector
        result = F.cleanup(query, memory)

        assert jnp.allclose(result, memory[0])


class TestBatchOperations:
    """Test batch operations using vmap."""

    def test_batch_bind_bsc(self):
        """Test batch BSC binding."""
        key = jax.random.PRNGKey(42)
        x = jax.random.bernoulli(key, 0.5, shape=(10, 10000))
        y = jax.random.bernoulli(key, 0.5, shape=(10, 10000))

        result = F.batch_bind_bsc(x, y)

        assert result.shape == (10, 10000)

    def test_batch_bind_map(self):
        """Test batch MAP binding."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, shape=(10, 10000))
        y = jax.random.normal(key, shape=(10, 10000))

        result = F.batch_bind_map(x, y)

        assert result.shape == (10, 10000)

    def test_batch_cosine_similarity(self):
        """Test batch cosine similarity computation."""
        key = jax.random.PRNGKey(42)
        queries = jax.random.normal(key, shape=(10, 10000))
        target = jax.random.normal(key, shape=(10000,))

        similarities = F.batch_cosine_similarity(queries, target)

        assert similarities.shape == (10,)
        assert jnp.all((similarities >= -1) & (similarities <= 1))


class TestHRROperations:
    """Test Holographic Reduced Representations operations."""

    def test_bind_hrr_inverse(self):
        """Test HRR binding inverse."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, shape=(10000,))
        y = jax.random.normal(key, shape=(10000,))

        # Normalize
        x = x / jnp.linalg.norm(x)
        y = y / jnp.linalg.norm(y)

        # bind(bind(x, y), inverse(y)) should be close to x
        bound = F.bind_hrr(x, y)
        y_inv = F.inverse_hrr(y)
        unbound = F.bind_hrr(bound, y_inv)

        # Normalize for comparison
        unbound = unbound / jnp.linalg.norm(unbound)

        similarity = jnp.sum(x * unbound)
        assert similarity > 0.8  # Should be quite similar

    def test_bundle_hrr_normalization(self):
        """Test that HRR bundling produces normalized vectors."""
        key = jax.random.PRNGKey(42)
        vectors = jax.random.normal(key, shape=(10, 10000))

        bundled = F.bundle_hrr(vectors, axis=0)
        norm = jnp.linalg.norm(bundled)

        assert jnp.allclose(norm, 1.0, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
