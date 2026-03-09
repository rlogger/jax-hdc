# Changelog

All notable changes to JAX-HDC will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- BSBC (Binary Sparse Block Codes) VSA model
- CGR (Cyclic Group Representation) VSA model
- MCR (Modular Composite Representation) VSA model
- VTB (Vector-Derived Transformation Binding) VSA model
- KernelEncoder (RBF kernel approximation via random Fourier features)
- GraphEncoder for graph structures
- LVQClassifier (Learning Vector Quantization)
- RegularizedLSClassifier (regularized least squares)
- SparseDistributedMemory, HopfieldMemory, and AttentionMemory modules
- Integration tests (end-to-end encode/train/predict)
- Performance benchmark suite
- `cleanup()` with `return_similarity` support
- Metrics module (`jax_hdc/metrics.py`)

### Changed
- Replaced `black`/`isort`/`flake8` with `ruff` for linting and formatting
- Removed `numpy` and `optax` from core dependencies
- Centralized JAX dataclass registration in `_compat.py`
- Reduced `utils.py` to `normalize` and `benchmark_function`

## [0.1.0-alpha] - 2024-11-03

### Added
- Core functional operations (bind, bundle, permute, similarity)
- Four VSA model implementations: BSC, MAP, HRR, FHRR
- Three encoder types: RandomEncoder, LevelEncoder, ProjectionEncoder
- Two classification models: CentroidClassifier, AdaptiveHDC
- Unit tests for core operations and VSA models
- Reference examples: basic operations, Kanerva's example, classification
- Documentation structure (Sphinx/ReadTheDocs ready)
- MIT License

[Unreleased]: https://github.com/rlogger/jax-hdc/compare/v0.1.0-alpha...HEAD
[0.1.0-alpha]: https://github.com/rlogger/jax-hdc/releases/tag/v0.1.0-alpha
