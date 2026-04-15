<p align="center">
    <a href="https://github.com/rlogger/jax-hdc/blob/main/LICENSE"><img alt="GitHub license" src="https://img.shields.io/badge/license-MIT-blue.svg?style=flat" /></a>
    <img alt="Development Status" src="https://img.shields.io/badge/status-alpha-orange.svg?style=flat" />
    <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-blue.svg?style=flat" />
</p>

# JAX-HDC

**A high-performance JAX library for Hyperdimensional Computing and Vector Symbolic Architectures**

JAX-HDC provides efficient implementations of Hyperdimensional Computing (HDC) and Vector Symbolic Architectures (VSA) using JAX. The library leverages XLA compilation, automatic vectorization, and hardware acceleration with a functional programming interface.

## Roadmap: Beyond TorchHD

> **We're not porting TorchHD to JAX. We're building the HDC library researchers actually publish with.**

Six things no HDC library offers today. Ship them, then submit to JMLR MLOSS.

### 1. Learnable codebooks, not random
Random codebooks are an 80s convenience, not a law. Every encoder in JAX-HDC learns its codebook end-to-end through the full VSA pipeline — bind, bundle, permute, cleanup — all differentiable.
- [ ] Straight-through estimators for BSC, BSBC, and CGR
- [ ] Backprop through every VSA primitive in `jax_hdc/vsa.py`
- [ ] First-class Flax, Equinox, and Optax interop
- [ ] Joint training with downstream neural layers (hybrid neuro-symbolic loops)

### 2. Probabilistic by default
Every hypervector carries a distribution, not a point. Calibrated uncertainty for classification and retrieval — a first in the HDC field.
- [ ] Bayesian hypervectors with posterior sampling
- [ ] Variational codebooks via reparameterization
- [ ] Conformal prediction for VSA classifiers
- [ ] Temperature-calibrated similarity for retrieval

### 3. TPU-scale hypervectors
JAX's distribution story, applied to HDC end-to-end. Million-dimensional vectors across pods, streaming ingest, memory-mapped billion-symbol codebooks.
- [ ] `pmap` and `shard_map` kernels for every VSA operation
- [ ] Zero-copy sharded codebooks across accelerators
- [ ] Online and streaming classifiers with concept-drift handling
- [ ] Memory-mapped codebooks for >1B-symbol vocabularies

### 4. Reasoning, not classification
HDC is a cognitive architecture — use it as one. First-class analogical mapping and knowledge-graph primitives, not just `CentroidClassifier.fit`.
- [ ] Structure-mapping engine (SME) on VSAs
- [ ] Knowledge-graph embeddings and link prediction
- [ ] Rule induction and program synthesis on hypervectors
- [ ] Compositional generalization benchmarks (SCAN, COGS, CLEVR-CoGenT)

### 5. Capacity, measured
Every HDC paper re-derives the same noise bounds. Ship them as a library.
- [ ] Capacity, crosstalk, and binding-noise probes per VSA model
- [ ] Analytic and empirical noise budgets
- [ ] Robustness and adversarial audits
- [ ] Interpretable bundle decomposition and cleanup tracing

### 6. A published benchmark
- [ ] Reproducible head-to-head vs. TorchHD on 15+ datasets
- [ ] Standardized accuracy, throughput, memory, and energy-per-inference reports
- [ ] Public leaderboard with seeded, containerized runs
- [ ] Paper draft targeting JMLR MLOSS

---

## Features

- XLA compilation and automatic kernel fusion through JAX
- Native GPU/TPU support
- Functional design compatible with JAX transformations (`jit`, `vmap`, `pmap`)
- Eight VSA models: BSC, MAP, HRR, FHRR, BSBC, CGR, MCR, VTB
- Encoders for discrete, continuous, kernel, and graph data
- Classification models: centroid, adaptive, LVQ, regularized least squares
- Memory modules: SDM, Hopfield networks, attention-based retrieval

## Installation

```bash
git clone https://github.com/rlogger/jax-hdc.git
cd jax-hdc
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

### Nix

```bash
nix develop   # flakes
nix-shell     # traditional
```

## Quick Start

```python
import jax
from jax_hdc import MAP, RandomEncoder, CentroidClassifier

model = MAP.create(dimensions=10000)
key = jax.random.PRNGKey(42)

# Bind and bundle
x = model.random(key, (10000,))
y = model.random(key, (10000,))
bound = model.bind(x, y)
bundled = model.bundle(jax.numpy.stack([x, y]), axis=0)

# Classification pipeline
encoder = RandomEncoder.create(
    num_features=20, num_values=10, dimensions=10000,
    vsa_model=model, key=key,
)
data = jax.random.randint(key, (100, 20), 0, 10)
labels = jax.random.randint(key, (100,), 0, 5)
encoded = encoder.encode_batch(data)

classifier = CentroidClassifier.create(
    num_classes=5, dimensions=10000, vsa_model=model,
)
classifier = classifier.fit(encoded, labels)
accuracy = classifier.score(encoded, labels)
```

## VSA Models

| Model | Description |
|-------|-------------|
| **BSC** | Binary Spatter Codes — XOR binding, majority bundling |
| **MAP** | Multiply-Add-Permute — element-wise multiply, normalized sum |
| **HRR** | Holographic Reduced Representations — circular convolution |
| **FHRR** | Fourier HRR — complex-valued, element-wise multiply |
| **BSBC** | Binary Sparse Block Codes — block-sparse binary |
| **CGR** | Cyclic Group Representation — modular addition binding |
| **MCR** | Modular Composite Representation — phasor arithmetic |
| **VTB** | Vector-Derived Transformation Binding — matrix multiplication |

All models share the same API: `bind`, `bundle`, `inverse`, `similarity`, `random`.

## Encoders

- **RandomEncoder** — discrete features via codebook lookup
- **LevelEncoder** — continuous values via level interpolation
- **ProjectionEncoder** — high-dimensional data via random projection
- **KernelEncoder** — RBF kernel approximation (Random Fourier Features)
- **GraphEncoder** — graph structures via node binding

## Classification Models

- **CentroidClassifier** — single-pass centroid prototypes
- **AdaptiveHDC** — iterative prototype refinement
- **LVQClassifier** — Learning Vector Quantization
- **RegularizedLSClassifier** — regularized least squares

## Memory Modules

- **SparseDistributedMemory** — content-addressable storage (Kanerva SDM)
- **HopfieldMemory** — modern Hopfield network with softmax attention
- **AttentionMemory** — scaled dot-product attention with multi-head support

## Development

```bash
pytest tests/ -v                              # run tests
pytest tests/ --cov=jax_hdc --cov-report=html # with coverage
ruff check jax_hdc/                           # lint
ruff format jax_hdc/                          # format
mypy jax_hdc/                                 # type check
```

## Examples

```bash
python examples/basic_operations.py      # core HDC operations
python examples/kanerva_example.py       # analogical reasoning
python examples/classification_simple.py # classification pipeline
```

## License

MIT — see [LICENSE](LICENSE).

## References

- Kanerva, P. (2009). "Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors"
- Plate, T. A. (1995). "Holographic Reduced Representations"
- Gayler, R. W. (2003). "Vector Symbolic Architectures answer Jackendoff's challenges for cognitive neuroscience"
