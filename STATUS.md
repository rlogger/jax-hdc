# JAX-HDC Development Status

**Version**: 0.1.0-alpha
**Status**: Alpha - Core functionality implemented, API subject to change
**Last Updated**: 2024-11-03

## Overview

JAX-HDC is in active development. This document provides transparency about what works, what doesn't, and what's planned.

## What Works Now

### Core Operations ✓
All core HDC operations are implemented and tested:
- Binary Spatter Code (BSC) operations
- Multiply-Add-Permute (MAP) operations
- Holographic Reduced Representations (HRR) operations
- Fourier HRR (FHRR) operations
- Universal operations (permute, cleanup)
- Batch operations via vmap

### VSA Models ✓
Four complete VSA models with unified interface:
- BSC (binary vectors, XOR binding)
- MAP (real vectors, element-wise multiplication)
- HRR (circular convolution via FFT)
- FHRR (complex vectors, efficient binding)

### Encoders ✓
Three encoder types for different data:
- RandomEncoder (discrete features)
- LevelEncoder (continuous values)
- ProjectionEncoder (high-dimensional data)

### Classification ✓
Two classification algorithms:
- CentroidClassifier (single-pass training, O(n) time)
- AdaptiveHDC (iterative refinement)

### Examples ✓
Three reference implementations:
- `examples/basic_operations.py` - Core operations demo
- `examples/kanerva_example.py` - Kanerva's "Dollar of Mexico"
- `examples/classification_simple.py` - End-to-end classification

### Testing ✓
Test coverage for critical paths:
- Functional operations (bind, bundle, permute, similarity)
- VSA models (all four implementations)
- Property verification (commutativity, invertibility, etc.)

## What's Not Ready Yet

### Missing Tests
- [ ] Encoder tests
- [ ] Classifier tests
- [ ] Integration tests
- [ ] Performance benchmarks

### Missing Documentation
- [ ] ReadTheDocs site (structure exists, not hosted)
- [ ] Tutorial notebooks
- [ ] API reference (generated from docstrings)
- [ ] Performance benchmarking methodology

### Missing Infrastructure
- [ ] PyPI package (not published)
- [ ] GitHub Actions CI (workflow defined, not tested)
- [ ] Code coverage reporting
- [ ] Automated releases

### Missing Features (Phase 2)
- [ ] Additional VSA models (B-SBC, CGR, MCR, VTB)
- [ ] Memory modules (SDM, Hopfield)
- [ ] Advanced learning (LVQ, regularized LS)
- [ ] Additional encoders (kernel, graph)

## Known Issues

1. **Installation**: Must install from source, not available on PyPI
2. **Documentation**: ReadTheDocs not configured/hosted
3. **Performance**: No formal benchmarks, claims are predictions
4. **API Stability**: Breaking changes possible in alpha
5. **Test Coverage**: Incomplete (core operations covered, encoders/models not)

## Verified Claims

✓ **Functional design**: All operations are pure functions
✓ **JAX compatibility**: Works with jit, vmap (pmap not tested)
✓ **Four VSA models**: BSC, MAP, HRR, FHRR implemented
✓ **Examples run**: All three examples execute successfully
✓ **Tests pass**: Core operations and VSA models tested

## Unverified Claims

⚠ **"10-100x speedups"**: Not benchmarked, predicted based on JAX characteristics
⚠ **"Comprehensive test coverage"**: Only partial coverage
⚠ **"Property-based verification"**: Standard unit tests, not property-based
⚠ **GPU/TPU performance**: Not formally benchmarked
⚠ **pmap compatibility**: Not tested
⚠ **grad compatibility**: Not tested (HDC operations not typically differentiable)

## Installation Verification

From source installation works:
```bash
git clone https://github.com/rlogger/jax-hdc.git
cd jax-hdc
pip install -e .
# ✓ Installs successfully
# ✓ Can import jax_hdc
# ✓ Examples run
# ✓ Tests pass
```

## API Stability

**Current Status**: Unstable

Modules likely to change:
- `embeddings.py` - Encoder API may evolve
- `models.py` - Classifier API may evolve
- `utils.py` - Utility functions may be reorganized

Modules likely stable:
- `functional.py` - Core operations unlikely to change
- `vsa.py` - VSA interface established

## Performance Expectations

**Predicted** (not benchmarked):
- 8-10x speedup for basic operations vs NumPy/PyTorch
- 10-20x speedup for encoding pipelines
- Actual performance depends on hardware and workload

**To verify**: Run formal benchmarks with methodology

## How to Help

High priority contributions:
1. Test coverage for encoders and classifiers
2. Performance benchmarking suite
3. ReadTheDocs configuration
4. GitHub Actions CI verification
5. Documentation improvements

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## Roadmap Timeline

**v0.1.0-alpha** (Current): Core functionality
**v0.2.0-beta** (Planned Q1 2025): Phase 2 features, tests, benchmarks
**v1.0.0** (Planned Q2 2025): Production ready, API stable, PyPI published

## Transparency Statement

This project is in early development. The README contains:
- **Verified**: Core implementation, examples, basic testing
- **Predicted**: Performance numbers (not benchmarked)
- **Planned**: Documentation, PyPI, advanced features

We prioritize transparency over marketing. See this document and [CHANGELOG.md](CHANGELOG.md) for accurate status.

## Questions?

Open an issue: https://github.com/rlogger/jax-hdc/issues
