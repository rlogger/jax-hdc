"""Performance regression tests for JAX-HDC models."""

import time
import jax
import jax.numpy as jnp
import pytest
from jax_hdc import MAP, CentroidClassifier, AdaptiveHDC

@pytest.mark.benchmark
def test_centroid_training_speed():
    """Ensure CentroidClassifier training is vectorized (O(N) -> O(1) ops)."""
    # Setup large synthetic problem
    N, D, C = 10000, 2000, 50
    key = jax.random.PRNGKey(42)
    
    model = MAP.create(dimensions=D)
    classifier = CentroidClassifier.create(num_classes=C, dimensions=D, vsa_model=model)
    
    # Fake data
    train_hvs = jnp.zeros((N, D))
    train_labels = jax.random.randint(key, (N,), 0, C)
    
    # Warmup
    classifier = classifier.fit(train_hvs[:100], train_labels[:100])
    
    # Benchmark
    start = time.perf_counter()
    classifier = classifier.fit(train_hvs, train_labels)
    # Block until ready is crucial for JAX timing
    classifier.prototypes.block_until_ready()
    end = time.perf_counter()
    
    duration = end - start
    print(f"\nTraining (N={N}, D={D}, C={C}) took {duration:.4f}s")
    
    # Threshold: Should be < 0.5s on most CPUs for this size if vectorized.
    # Non-vectorized loop would take > 2.0s
    assert duration < 1.0, f"Training too slow ({duration:.4f}s), vectorization might be broken"

@pytest.mark.benchmark
def test_adaptive_training_speed():
    """Ensure AdaptiveHDC uses jax.lax.scan."""
    N, D, C = 1000, 2000, 10
    key = jax.random.PRNGKey(42)
    
    model = MAP.create(dimensions=D)
    classifier = AdaptiveHDC.create(num_classes=C, dimensions=D, vsa_model=model)
    
    train_hvs = jnp.zeros((N, D))
    train_labels = jax.random.randint(key, (N,), 0, C)
    
    # Warmup
    classifier = classifier.fit(train_hvs[:10], train_labels[:10], epochs=1)
    
    # Benchmark
    start = time.perf_counter()
    classifier = classifier.fit(train_hvs, train_labels, epochs=5)
    classifier.prototypes.block_until_ready()
    end = time.perf_counter()
    
    duration = end - start
    print(f"\nAdaptive Training (N={N}, Epochs=5) took {duration:.4f}s")
    
    # Threshold checks if JIT compilation worked effectively
    assert duration < 2.0, "Adaptive training too slow, check jax.lax.scan implementation"
