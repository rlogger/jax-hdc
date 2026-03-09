"""Shared pytest fixtures for JAX-HDC tests."""

import jax
import jax.numpy as jnp
import pytest

from jax_hdc import BSC, FHRR, HRR, MAP
from jax_hdc.embeddings import LevelEncoder, ProjectionEncoder, RandomEncoder
from jax_hdc.models import AdaptiveHDC, CentroidClassifier


@pytest.fixture
def prng_key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def default_dimensions():
    return 100


@pytest.fixture
def map_model(default_dimensions):
    return MAP.create(dimensions=default_dimensions)


@pytest.fixture
def bsc_model(default_dimensions):
    return BSC.create(dimensions=default_dimensions)


@pytest.fixture
def hrr_model(default_dimensions):
    return HRR.create(dimensions=default_dimensions)


@pytest.fixture
def fhrr_model(default_dimensions):
    return FHRR.create(dimensions=default_dimensions)


@pytest.fixture
def random_encoder(default_dimensions, prng_key):
    return RandomEncoder.create(
        num_features=5, num_values=10, dimensions=default_dimensions, vsa_model="map", key=prng_key
    )


@pytest.fixture
def level_encoder(default_dimensions, prng_key):
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
    return ProjectionEncoder.create(
        input_dim=50, dimensions=default_dimensions, vsa_model="map", key=prng_key
    )


@pytest.fixture
def centroid_classifier(default_dimensions, prng_key):
    return CentroidClassifier.create(
        num_classes=3, dimensions=default_dimensions, vsa_model="map", key=prng_key
    )


@pytest.fixture
def adaptive_hdc_classifier(default_dimensions, prng_key):
    return AdaptiveHDC.create(
        num_classes=3, dimensions=default_dimensions, vsa_model="map", key=prng_key
    )


@pytest.fixture
def classification_data(map_model, prng_key):
    key_train, key_test = jax.random.split(prng_key)
    train_hvs = map_model.random(key_train, (30, map_model.dimensions))
    train_labels = jnp.array([0] * 10 + [1] * 10 + [2] * 10)
    test_hvs = map_model.random(key_test, (15, map_model.dimensions))
    test_labels = jnp.array([0] * 5 + [1] * 5 + [2] * 5)
    return {
        "train_hvs": train_hvs,
        "train_labels": train_labels,
        "test_hvs": test_hvs,
        "test_labels": test_labels,
    }
