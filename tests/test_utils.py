"""Tests for utility functions."""

import os

import jax
import jax.numpy as jnp
import pytest

from jax_hdc import MAP, CentroidClassifier, RandomEncoder, utils


class TestConfigureMemory:
    """Tests for configure_memory."""

    def test_configure_memory_defaults(self):
        """Test memory configuration with defaults."""
        utils.configure_memory(preallocate=False, memory_fraction=0.8)

        assert os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE") == "false"
        assert os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION") == "0.8"

    def test_configure_memory_custom(self):
        """Test memory configuration with custom values."""
        utils.configure_memory(preallocate=True, memory_fraction=0.5, device="gpu")

        assert os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE") == "true"
        assert os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION") == "0.5"

    def test_configure_memory_cpu(self):
        """Test memory configuration for CPU."""
        utils.configure_memory(device="cpu")

        # Should not raise any errors
        assert True


class TestGetDevice:
    """Tests for get_device."""

    def test_get_default_device(self):
        """Test getting default CPU device."""
        device = utils.get_device("cpu", 0)

        assert device is not None
        assert device.platform == "cpu"

    def test_get_device_fallback(self):
        """Test device fallback when GPU not available."""
        # Try to get a device that might not exist
        # Should fallback to CPU without error
        device = utils.get_device("tpu", 0)

        assert device is not None


class TestGetDeviceMemoryStats:
    """Tests for get_device_memory_stats."""

    def test_get_memory_stats(self):
        """Test getting memory statistics."""
        stats = utils.get_device_memory_stats()

        assert isinstance(stats, dict)
        # Either has memory stats or an error message
        assert len(stats) > 0

    def test_get_memory_stats_specific_device(self):
        """Test getting memory statistics for specific device."""
        device = jax.devices("cpu")[0]
        stats = utils.get_device_memory_stats(device)

        assert isinstance(stats, dict)


class TestSetRandomSeed:
    """Tests for set_random_seed."""

    def test_set_random_seed(self):
        """Test setting random seed."""
        key = utils.set_random_seed(42)

        assert isinstance(key, jax.Array)
        assert key.shape == (2,)  # JAX PRNG keys have shape (2,)

    def test_set_random_seed_reproducibility(self):
        """Test that same seed produces same key."""
        key1 = utils.set_random_seed(42)
        key2 = utils.set_random_seed(42)

        assert jnp.array_equal(key1, key2)

    def test_different_seeds_produce_different_keys(self):
        """Test that different seeds produce different keys."""
        key1 = utils.set_random_seed(42)
        key2 = utils.set_random_seed(43)

        assert not jnp.array_equal(key1, key2)


class TestBenchmarkFunction:
    """Tests for benchmark_function."""

    def test_benchmark_simple_function(self):
        """Test benchmarking a simple function."""

        def add_arrays(x, y):
            return x + y

        x = jnp.ones(1000)
        y = jnp.ones(1000)

        stats = utils.benchmark_function(add_arrays, x, y, num_trials=10, warmup=2)

        assert "mean_ms" in stats
        assert "std_ms" in stats
        assert "min_ms" in stats
        assert "max_ms" in stats
        assert "median_ms" in stats
        assert stats["num_trials"] == 10

        # All times should be positive
        assert stats["mean_ms"] >= 0
        assert stats["min_ms"] >= 0
        assert stats["max_ms"] >= stats["min_ms"]

    def test_benchmark_with_kwargs(self):
        """Test benchmarking with keyword arguments."""

        def multiply_with_factor(x, factor=2.0):
            return x * factor

        x = jnp.ones(1000)

        stats = utils.benchmark_function(multiply_with_factor, x, factor=3.0, num_trials=5)

        assert stats["num_trials"] == 5
        assert stats["mean_ms"] >= 0


class TestCheckShapes:
    """Tests for check_shapes."""

    def test_check_shapes_same(self):
        """Test checking shapes of same-shaped arrays."""
        x = jnp.ones((100, 10))
        y = jnp.zeros((100, 10))

        # Should not raise
        utils.check_shapes(x, y)

    def test_check_shapes_different(self):
        """Test checking shapes of different-shaped arrays."""
        x = jnp.ones((100, 10))
        y = jnp.zeros((50, 10))

        with pytest.raises(ValueError, match="Shape mismatch"):
            utils.check_shapes(x, y)

    def test_check_shapes_with_expected_ndim(self):
        """Test checking shapes with expected dimensions."""
        x = jnp.ones((100, 10))
        y = jnp.zeros((100, 10))

        # Should not raise
        utils.check_shapes(x, y, expected_ndim=2)

    def test_check_shapes_wrong_ndim(self):
        """Test checking shapes with wrong dimensions."""
        x = jnp.ones((100, 10))

        with pytest.raises(ValueError, match="dimensions"):
            utils.check_shapes(x, expected_ndim=1)

    def test_check_shapes_empty(self):
        """Test checking shapes with no arrays."""
        # Should not raise
        utils.check_shapes()


class TestNormalize:
    """Tests for normalize."""

    def test_normalize_1d(self):
        """Test normalizing a 1D vector."""
        x = jnp.array([3.0, 4.0])
        normalized = utils.normalize(x)

        norm = jnp.linalg.norm(normalized)
        assert jnp.allclose(norm, 1.0)

    def test_normalize_2d(self):
        """Test normalizing 2D array along last axis."""
        x = jax.random.normal(jax.random.PRNGKey(0), (10, 100))
        normalized = utils.normalize(x, axis=-1)

        norms = jnp.linalg.norm(normalized, axis=-1)
        assert jnp.allclose(norms, 1.0)

    def test_normalize_custom_axis(self):
        """Test normalizing along custom axis."""
        x = jax.random.normal(jax.random.PRNGKey(0), (10, 100))
        normalized = utils.normalize(x, axis=0)

        norms = jnp.linalg.norm(normalized, axis=0)
        assert jnp.allclose(norms, 1.0)

    def test_normalize_zero_vector(self):
        """Test normalizing zero vector with eps."""
        x = jnp.zeros(10)
        normalized = utils.normalize(x, eps=1e-8)

        # Should not produce NaN
        assert not jnp.any(jnp.isnan(normalized))


class TestPrintModelInfo:
    """Tests for print_model_info."""

    def test_print_vsa_model_info(self, capsys):
        """Test printing VSA model info."""
        model = MAP.create(dimensions=100)
        utils.print_model_info(model)

        captured = capsys.readouterr()
        assert "MAP" in captured.out
        assert "dimensions" in captured.out

    def test_print_encoder_info(self, capsys):
        """Test printing encoder info."""
        encoder = RandomEncoder.create(
            num_features=5, num_values=10, dimensions=100, key=jax.random.PRNGKey(42)
        )
        utils.print_model_info(encoder)

        captured = capsys.readouterr()
        assert "RandomEncoder" in captured.out

    def test_print_classifier_info(self, capsys):
        """Test printing classifier info."""
        classifier = CentroidClassifier.create(
            num_classes=3, dimensions=100, key=jax.random.PRNGKey(42)
        )
        utils.print_model_info(classifier)

        captured = capsys.readouterr()
        assert "CentroidClassifier" in captured.out


class TestCountParameters:
    """Tests for count_parameters."""

    def test_count_vsa_parameters(self):
        """Test counting VSA model parameters."""
        model = MAP.create(dimensions=100)
        count = utils.count_parameters(model)

        # MAP has no stored parameters (only dimensions metadata)
        assert count == 0

    def test_count_encoder_parameters(self):
        """Test counting encoder parameters."""
        encoder = RandomEncoder.create(
            num_features=5, num_values=10, dimensions=100, key=jax.random.PRNGKey(42)
        )
        count = utils.count_parameters(encoder)

        # Codebook has shape (5, 10, 100) = 5000 parameters
        assert count == 5 * 10 * 100

    def test_count_classifier_parameters(self):
        """Test counting classifier parameters."""
        classifier = CentroidClassifier.create(
            num_classes=3, dimensions=100, key=jax.random.PRNGKey(42)
        )
        count = utils.count_parameters(classifier)

        # Prototypes have shape (3, 100) = 300 parameters
        assert count == 3 * 100


class TestToDevice:
    """Tests for to_device."""

    def test_to_device_single_array(self):
        """Test moving single array to device."""
        device = jax.devices("cpu")[0]
        x = jnp.ones(100)

        x_device = utils.to_device(x, device)

        assert isinstance(x_device, jax.Array)

    def test_to_device_tuple(self):
        """Test moving tuple of arrays to device."""
        device = jax.devices("cpu")[0]
        x = jnp.ones(100)
        y = jnp.zeros(100)

        result = utils.to_device((x, y), device)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_to_device_list(self):
        """Test moving list of arrays to device."""
        device = jax.devices("cpu")[0]
        x = jnp.ones(100)
        y = jnp.zeros(100)

        result = utils.to_device([x, y], device)

        assert isinstance(result, list)
        assert len(result) == 2


class TestCheckNanInf:
    """Tests for check_nan_inf."""

    def test_check_valid_array(self):
        """Test checking valid array."""
        x = jnp.array([1.0, 2.0, 3.0])

        # Should not raise
        utils.check_nan_inf(x)

    def test_check_nan_array(self):
        """Test checking array with NaN."""
        x = jnp.array([1.0, jnp.nan, 3.0])

        with pytest.raises(ValueError, match="NaN"):
            utils.check_nan_inf(x)

    def test_check_inf_array(self):
        """Test checking array with Inf."""
        x = jnp.array([1.0, jnp.inf, 3.0])

        with pytest.raises(ValueError, match="Inf"):
            utils.check_nan_inf(x)

    def test_check_with_custom_name(self):
        """Test checking with custom name."""
        x = jnp.array([1.0, jnp.nan, 3.0])

        with pytest.raises(ValueError, match="my_array"):
            utils.check_nan_inf(x, name="my_array")


class TestGetVersionInfo:
    """Tests for get_version_info."""

    def test_get_version_info(self):
        """Test getting version information."""
        info = utils.get_version_info()

        assert isinstance(info, dict)
        assert "jax_hdc" in info
        assert "jax" in info
        assert "jaxlib" in info

        # JAX-HDC version should be a string
        assert isinstance(info["jax_hdc"], str)
        assert isinstance(info["jax"], str)

    def test_version_format(self):
        """Test that version strings have expected format."""
        info = utils.get_version_info()

        # Version should contain something like "0.1.0"
        assert len(info["jax_hdc"]) > 0
