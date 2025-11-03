"""Tests for VSA models."""

import pytest
import jax
import jax.numpy as jnp
from jax_hdc.vsa import BSC, MAP, HRR, FHRR, create_vsa_model


class TestVSAModels:
    """Test VSA model creation and basic operations."""

    def test_create_bsc(self):
        """Test BSC model creation."""
        model = BSC.create(dimensions=10000)

        assert model.name == "bsc"
        assert model.dimensions == 10000

    def test_create_map(self):
        """Test MAP model creation."""
        model = MAP.create(dimensions=10000)

        assert model.name == "map"
        assert model.dimensions == 10000

    def test_create_hrr(self):
        """Test HRR model creation."""
        model = HRR.create(dimensions=10000)

        assert model.name == "hrr"
        assert model.dimensions == 10000

    def test_create_fhrr(self):
        """Test FHRR model creation."""
        model = FHRR.create(dimensions=10000)

        assert model.name == "fhrr"
        assert model.dimensions == 10000

    def test_factory_function(self):
        """Test create_vsa_model factory function."""
        for model_type in ["bsc", "map", "hrr", "fhrr"]:
            model = create_vsa_model(model_type, dimensions=10000)
            assert model.name == model_type
            assert model.dimensions == 10000

    def test_factory_invalid_type(self):
        """Test that invalid model type raises error."""
        with pytest.raises(ValueError):
            create_vsa_model("invalid_model", dimensions=10000)


class TestBSCModel:
    """Test BSC model operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = BSC.create(dimensions=10000)
        self.key = jax.random.PRNGKey(42)

    def test_random_generation(self):
        """Test random hypervector generation."""
        x = self.model.random(self.key, (10000,))

        assert x.shape == (10000,)
        assert x.dtype == jnp.bool_
        # Should be roughly 50% ones
        assert 0.45 < jnp.mean(x) < 0.55

    def test_bind_operation(self):
        """Test binding operation."""
        x = self.model.random(self.key, (10000,))
        y = self.model.random(self.key, (10000,))

        bound = self.model.bind(x, y)

        assert bound.shape == (10000,)
        assert bound.dtype == jnp.bool_

    def test_bundle_operation(self):
        """Test bundling operation."""
        vectors = self.model.random(self.key, (10, 10000))

        bundled = self.model.bundle(vectors, axis=0)

        assert bundled.shape == (10000,)
        assert bundled.dtype == jnp.bool_

    def test_inverse_operation(self):
        """Test inverse operation."""
        x = self.model.random(self.key, (10000,))
        x_inv = self.model.inverse(x)

        # For BSC, inverse is identity
        assert jnp.allclose(x_inv, x)

    def test_similarity_computation(self):
        """Test similarity computation."""
        x = self.model.random(self.key, (10000,))
        y = self.model.random(self.key, (10000,))

        sim = self.model.similarity(x, y)

        assert 0 <= sim <= 1


class TestMAPModel:
    """Test MAP model operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = MAP.create(dimensions=10000)
        self.key = jax.random.PRNGKey(42)

    def test_random_generation(self):
        """Test random hypervector generation."""
        x = self.model.random(self.key, (10000,))

        assert x.shape == (10000,)
        # Should be normalized
        norm = jnp.linalg.norm(x)
        assert jnp.allclose(norm, 1.0)

    def test_bind_operation(self):
        """Test binding operation."""
        x = self.model.random(self.key, (10000,))
        y = self.model.random(self.key, (10000,))

        bound = self.model.bind(x, y)

        assert bound.shape == (10000,)

    def test_bundle_operation(self):
        """Test bundling operation."""
        vectors = self.model.random(self.key, (10, 10000))

        bundled = self.model.bundle(vectors, axis=0)

        assert bundled.shape == (10000,)
        # Should be normalized
        norm = jnp.linalg.norm(bundled)
        assert jnp.allclose(norm, 1.0, atol=1e-6)

    def test_inverse_operation(self):
        """Test inverse operation."""
        x = self.model.random(self.key, (10000,))
        y = self.model.random(self.key, (10000,))

        bound = self.model.bind(x, y)
        x_inv = self.model.inverse(x)
        unbound = self.model.bind(bound, x_inv)

        # Should get back something close to y
        sim = self.model.similarity(unbound / jnp.linalg.norm(unbound), y)
        assert sim > 0.5

    def test_similarity_computation(self):
        """Test similarity computation."""
        x = self.model.random(self.key, (10000,))

        # Self-similarity should be 1
        sim = self.model.similarity(x, x)
        assert jnp.allclose(sim, 1.0, atol=1e-6)


class TestHRRModel:
    """Test HRR model operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = HRR.create(dimensions=10000)
        self.key = jax.random.PRNGKey(42)

    def test_random_generation(self):
        """Test random hypervector generation."""
        x = self.model.random(self.key, (10000,))

        assert x.shape == (10000,)
        norm = jnp.linalg.norm(x)
        assert jnp.allclose(norm, 1.0)

    def test_bind_unbind_cycle(self):
        """Test bind-unbind cycle."""
        x = self.model.random(self.key, (10000,))
        y = self.model.random(self.key, (10000,))

        bound = self.model.bind(x, y)
        y_inv = self.model.inverse(y)
        unbound = self.model.bind(bound, y_inv)

        # Normalize for comparison
        unbound = unbound / jnp.linalg.norm(unbound)

        sim = self.model.similarity(unbound, x)
        assert sim > 0.7  # Should be fairly similar


class TestFHRRModel:
    """Test FHRR model operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = FHRR.create(dimensions=10000)
        self.key = jax.random.PRNGKey(42)

    def test_random_generation(self):
        """Test random hypervector generation."""
        x = self.model.random(self.key, (10000,))

        assert x.shape == (10000,)
        assert x.dtype == jnp.complex64 or x.dtype == jnp.complex128
        # Should be on unit circle
        magnitudes = jnp.abs(x)
        assert jnp.allclose(magnitudes, 1.0, atol=1e-6)

    def test_bind_operation(self):
        """Test binding operation."""
        x = self.model.random(self.key, (10000,))
        y = self.model.random(self.key, (10000,))

        bound = self.model.bind(x, y)

        assert bound.shape == (10000,)
        # Should maintain unit magnitude
        magnitudes = jnp.abs(bound)
        assert jnp.allclose(magnitudes, 1.0, atol=1e-6)

    def test_inverse_conjugate(self):
        """Test that inverse is complex conjugate."""
        x = self.model.random(self.key, (10000,))
        x_inv = self.model.inverse(x)

        # Inverse should be conjugate
        assert jnp.allclose(x_inv, jnp.conj(x))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
