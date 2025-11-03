# Changelog

All notable changes to JAX-HDC will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned for v0.2.0-beta
- Complete test coverage for encoders and classifiers
- Performance benchmarking suite with reproducible methodology
- ReadTheDocs hosting
- GitHub Actions CI/CD
- Additional VSA models (B-SBC, CGR, MCR, VTB)
- Memory modules (SDM, Hopfield)

### Planned for v1.0.0
- PyPI package publication
- API stability guarantees
- Comprehensive documentation
- Production-ready performance validation

## [0.1.0-alpha] - 2024-11-03

### Added
- Core functional operations (bind, bundle, permute, similarity)
- Four VSA model implementations: BSC, MAP, HRR, FHRR
- Three encoder types: RandomEncoder, LevelEncoder, ProjectionEncoder
- Two classification models: CentroidClassifier, AdaptiveHDC
- Utility functions for device management and benchmarking
- Unit tests for core operations and VSA models
- Reference examples: basic operations, Kanerva's example, classification
- Documentation structure (Sphinx/ReadTheDocs ready)
- MIT License
- Development tooling (pytest, black, mypy, isort, flake8)

### Notes
- **Alpha release** - API subject to change
- Not published to PyPI (install from source only)
- Documentation hosted in repository (ReadTheDocs pending)
- Performance claims are predictions, not benchmarked
- Test coverage incomplete (core operations covered)

### Known Limitations
- No encoder or classifier tests
- No integration tests
- No performance benchmarks
- ReadTheDocs not configured
- CI/CD not activated

## Version History

- **0.1.0-alpha** (2024-11-03): Initial alpha release

[Unreleased]: https://github.com/rlogger/jax-hdc/compare/v0.1.0-alpha...HEAD
[0.1.0-alpha]: https://github.com/rlogger/jax-hdc/releases/tag/v0.1.0-alpha
