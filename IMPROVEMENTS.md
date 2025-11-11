# JAX-HDC Improvement Roadmap

## Quick Wins (1-2 hours each)

### 1. Constants Module
**Issue:** Epsilon value `1e-8` hardcoded in 8+ places
**Fix:**
```python
# jax_hdc/constants.py
DEFAULT_EPS = 1e-8
DEFAULT_DIMENSIONS = 10000
```
Then import and use: `from jax_hdc.constants import DEFAULT_EPS`

### 2. Better Error Messages
**Add validation:**
```python
def _validate_dimensions(x, y):
    if x.shape[-1] != y.shape[-1]:
        raise ValueError(
            f"Dimension mismatch: x has {x.shape[-1]} dims, "
            f"y has {y.shape[-1]} dims"
        )
```

### 3. Logging
**Add progress logging for long operations:**
```python
import logging
logger = logging.getLogger(__name__)

def fit(self, train_hvs, train_labels, epochs=10):
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        # ... training code
```

---

## Performance Optimizations (2-4 hours each)

### 4. Vectorize CentroidClassifier.fit
**Current:** Python loop (models.py:214-246)
**Optimized:**
```python
# Use einsum for all classes at once
one_hot = jax.nn.one_hot(train_labels, self.num_classes)
weighted_hvs = jnp.einsum('nc,nd->cd', one_hot, train_hvs)
counts = jnp.sum(one_hot, axis=0)
centroids = weighted_hvs / counts[:, None]
```
**Speedup:** 5-10x for many classes

### 5. JIT-Compiled Training Loops
**Add @jax.jit to:**
- `AdaptiveHDC.fit` (currently not JIT-friendly)
- `CentroidClassifier.update_online`
- All batch operations

### 6. Mixed Precision Support
```python
# jax_hdc/config.py
class Config:
    dtype = jnp.float32  # or jnp.float16 for speed

# Use throughout:
vectors = jax.random.normal(key, shape).astype(Config.dtype)
```

---

## New Features (4-8 hours each)

### 7. Additional VSA Models (from roadmap)
**Priority order:**
1. **B-SBC** (Binary Sparse Block Codes) - memory efficient
2. **VTB** (Vector-Derived Transformation Binding) - good for sequences
3. **CGR** (Cyclic Group Representation) - theoretical interest

### 8. Memory Modules
```python
# jax_hdc/memory.py
class SparseDistributedMemory:
    """Kanerva's SDM for content-addressable memory"""

class ModernHopfield:
    """Modern Hopfield networks for HDC"""
```

### 9. Advanced Encoders
```python
# jax_hdc/embeddings.py
class TemporalEncoder:
    """For time series with sliding windows"""

class GraphEncoder:
    """For graph structures (nodes + edges)"""

class ImageEncoder:
    """2D spatial encoding with position"""
```

### 10. Model Persistence
```python
# jax_hdc/io.py
def save_model(model, path):
    """Save model to disk (using pickle/joblib)"""

def load_model(path):
    """Load model from disk"""
```

---

## Testing Improvements (2-3 hours)

### 11. Property-Based Testing
```python
# tests/test_properties.py
import hypothesis
from hypothesis import given, strategies as st

@given(st.integers(min_value=100, max_value=10000))
def test_random_vectors_orthogonal(dimensions):
    """Random vectors should be approximately orthogonal"""
    model = MAP.create(dimensions=dimensions)
    x = model.random(key, (dimensions,))
    y = model.random(key2, (dimensions,))
    sim = model.similarity(x, y)
    assert abs(sim) < 0.2  # Nearly orthogonal
```

### 12. Integration Tests
```python
# tests/test_integration.py
def test_end_to_end_classification():
    """Test full pipeline: encode → train → predict"""

def test_model_serialization():
    """Test save/load preserves behavior"""
```

### 13. Performance Regression Tests
```python
# tests/test_performance.py
@pytest.mark.benchmark
def test_binding_speed(benchmark):
    """Ensure binding stays fast"""
    benchmark(model.bind, x, y)
```

---

## CI/CD Enhancements (1-2 hours)

### 14. GitHub Actions Improvements
```yaml
# .github/workflows/ci.yml
jobs:
  test:
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11', '3.12']
        jax-version: ['0.4.20', 'latest']

  benchmark:
    # Run benchmarks and comment on PRs

  docs:
    # Build and deploy docs to GitHub Pages
```

### 15. Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.0.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.0.0
    hooks:
      - id: mypy
```

---

## Documentation (3-5 hours)

### 16. Tutorials
```markdown
docs/tutorials/
├── 01_getting_started.md
├── 02_understanding_vsa.md
├── 03_building_classifiers.md
├── 04_custom_encoders.md
└── 05_performance_tuning.md
```

### 17. API Reference
**Auto-generate from docstrings:**
```python
# All functions already have good docstrings!
# Just need to run sphinx-apidoc
```

### 18. Contribution Guide
**Enhance CONTRIBUTING.md:**
- Code style guide
- How to add new VSA models
- How to add new encoders
- Testing requirements

---

## Community/Project Health (1-2 hours each)

### 19. Issue Templates
```markdown
.github/ISSUE_TEMPLATE/
├── bug_report.md
├── feature_request.md
└── question.md
```

### 20. GitHub Actions Badges
```markdown
# README.md
![Tests](https://github.com/rlogger/jax-hdc/workflows/tests/badge.svg)
![Coverage](https://img.shields.io/codecov/c/github/rlogger/jax-hdc)
![PyPI](https://img.shields.io/pypi/v/jax-hdc)
```

### 21. PyPI Release
```bash
# Setup for PyPI
python -m build
python -m twine upload dist/*
```

---

## Research/Advanced (8+ hours)

### 22. Neural-HDC Hybrids
```python
# jax_hdc/neural.py
class NeuralEncoder(nn.Module):
    """Learn encoders with backprop"""

class HybridClassifier:
    """Combine HDC with neural networks"""
```

### 23. Distributed Training
```python
# Use pmap for multi-GPU
@jax.pmap
def train_step(state, batch):
    ...
```

### 24. Benchmark Suite vs TorchHD
**Compare:**
- Speed (operations/sec)
- Memory usage
- Classification accuracy
- Training time

---

## Priority Ranking

### Must Have (Before v1.0)
1. ✅ All tests passing (DONE)
2. ✅ CI/CD working (DONE)
3. 🔲 Complete documentation
4. 🔲 Real benchmarks
5. 🔲 PyPI release

### Should Have (v1.1)
6. 🔲 More examples (MNIST, real datasets)
7. 🔲 Model persistence
8. 🔲 Performance optimizations
9. 🔲 Pre-commit hooks

### Nice to Have (v1.2+)
10. 🔲 Additional VSA models
11. 🔲 Memory modules
12. 🔲 Neural-HDC hybrids
13. 🔲 Distributed training

---

## Quick Start: Weekend Projects

### Project 1: Documentation Sprint (4 hours)
```bash
cd docs/
sphinx-apidoc -f -o . ../jax_hdc/
# Edit quickstart.rst with real examples
make html
```

### Project 2: MNIST Example (3 hours)
```python
# examples/mnist_classification.py
# Show real accuracy on real dataset
# Compare BSC vs MAP vs HRR
```

### Project 3: Benchmarking (2 hours)
```python
# benchmarks/compare_models.py
# Measure actual speeds
# Update README with real numbers
```

### Project 4: Constants Refactor (1 hour)
```python
# Create jax_hdc/constants.py
# Replace all hardcoded 1e-8
# Add DEFAULT_DIMENSIONS
```

---

## Getting Help

- **Easy issues:** Documentation, examples, constants
- **Medium issues:** New encoders, benchmarks
- **Hard issues:** New VSA models, distributed training

Start with documentation and examples - high impact, low risk!
