"""Shared pytest fixtures for JAX-HDC tests."""

import jax
import jax.numpy as jnp
import pytest

from jax_hdc import BSC, FHRR, HRR, MAP
from jax_hdc.embeddings import LevelEncoder, ProjectionEncoder, RandomEncoder
from jax_hdc.models import AdaptiveHDC, CentroidClassifier


@pytest.fixture
def random_seed():
    """Default random seed for tests."""
    return 42


@pytest.fixture
def prng_key(random_seed):
    """JAX PRNG key for reproducible tests."""
    return jax.random.PRNGKey(random_seed)


@pytest.fixture
def default_dimensions():
    """Default hypervector dimensions for tests."""
    return 100


@pytest.fixture
def large_dimensions():
    """Large hypervector dimensions for similarity tests."""
    return 1000


# VSA Model Fixtures
@pytest.fixture
def map_model(default_dimensions):
    """Create a MAP VSA model."""
    return MAP.create(dimensions=default_dimensions)


@pytest.fixture
def bsc_model(default_dimensions):
    """Create a BSC VSA model."""
    return BSC.create(dimensions=default_dimensions)


@pytest.fixture
def hrr_model(default_dimensions):
    """Create an HRR VSA model."""
    return HRR.create(dimensions=default_dimensions)


@pytest.fixture
def fhrr_model(default_dimensions):
    """Create an FHRR VSA model."""
    return FHRR.create(dimensions=default_dimensions)


# Random Vector Fixtures
@pytest.fixture
def random_vectors_1d(prng_key, default_dimensions):
    """Generate a pair of random 1D vectors."""
    key_x, key_y = jax.random.split(prng_key)
    x = jax.random.normal(key_x, (default_dimensions,))
    y = jax.random.normal(key_y, (default_dimensions,))
    return x, y


@pytest.fixture
def random_vectors_2d(prng_key, default_dimensions):
    """Generate a batch of random vectors."""
    return jax.random.normal(prng_key, (10, default_dimensions))


@pytest.fixture
def random_binary_vectors_1d(prng_key, default_dimensions):
    """Generate a pair of random binary vectors."""
    key_x, key_y = jax.random.split(prng_key)
    x = jax.random.bernoulli(key_x, 0.5, (default_dimensions,))
    y = jax.random.bernoulli(key_y, 0.5, (default_dimensions,))
    return x, y


# Encoder Fixtures
@pytest.fixture
def random_encoder(default_dimensions, prng_key):
    """Create a RandomEncoder."""
    return RandomEncoder.create(
        num_features=5, num_values=10, dimensions=default_dimensions, vsa_model="map", key=prng_key
    )


@pytest.fixture
def level_encoder(default_dimensions, prng_key):
    """Create a LevelEncoder."""
    return LevelEncoder.create(
        num_levels=50,
        dimensions=default_dimensions,
        min_value=0.0,
        max_value=1.0,
        vsa_model="map",
        key=prng_key,
    )


@pytest.fixture
def projection_encoder(default_dimensions, prng_key):
    """Create a ProjectionEncoder."""
    return ProjectionEncoder.create(
        input_dim=50, dimensions=default_dimensions, vsa_model="map", key=prng_key
    )


# Classifier Fixtures
@pytest.fixture
def centroid_classifier(default_dimensions, prng_key):
    """Create a CentroidClassifier."""
    return CentroidClassifier.create(
        num_classes=3, dimensions=default_dimensions, vsa_model="map", key=prng_key
    )


@pytest.fixture
def adaptive_hdc_classifier(default_dimensions, prng_key):
    """Create an AdaptiveHDC classifier."""
    return AdaptiveHDC.create(
        num_classes=3, dimensions=default_dimensions, vsa_model="map", key=prng_key
    )


# Training Data Fixtures
@pytest.fixture
def classification_data(map_model, prng_key):
    """Create synthetic classification data."""
    key_train, key_test = jax.random.split(prng_key)

    # Training data
    train_hvs = map_model.random(key_train, (30, map_model.dimensions))
    train_labels = jnp.array([0] * 10 + [1] * 10 + [2] * 10)

    # Test data
    test_hvs = map_model.random(key_test, (15, map_model.dimensions))
    test_labels = jnp.array([0] * 5 + [1] * 5 + [2] * 5)

    return {
        "train_hvs": train_hvs,
        "train_labels": train_labels,
        "test_hvs": test_hvs,
        "test_labels": test_labels,
    }


# Device Fixtures
@pytest.fixture
def cpu_device():
    """Get CPU device."""
    return jax.devices("cpu")[0]


# Helper function fixtures
@pytest.fixture
def assert_normalized():
    """Helper function to assert vectors are normalized."""

    def _assert_normalized(vectors, axis=-1, atol=1e-5):
        norms = jnp.linalg.norm(vectors, axis=axis)
        assert jnp.allclose(norms, 1.0, atol=atol), f"Vectors not normalized: {norms}"

    return _assert_normalized


@pytest.fixture
def assert_similar():
    """Helper function to assert vectors are similar."""

    def _assert_similar(v1, v2, threshold=0.9):
        similarity = jnp.dot(v1, v2) / (jnp.linalg.norm(v1) * jnp.linalg.norm(v2))
        assert similarity > threshold, f"Similarity {similarity} < {threshold}"

    return _assert_similar


@pytest.fixture
def assert_dissimilar():
    """Helper function to assert vectors are dissimilar."""

    def _assert_dissimilar(v1, v2, threshold=0.5):
        similarity = jnp.dot(v1, v2) / (jnp.linalg.norm(v1) * jnp.linalg.norm(v2))
        assert similarity < threshold, f"Similarity {similarity} >= {threshold}"

    return _assert_dissimilar
