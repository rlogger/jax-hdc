# JAX-HDC Examples

Reference implementations demonstrating JAX-HDC capabilities with educational focus.

## Installation

Install JAX-HDC with example dependencies:

```bash
# From repository root
pip install -e ".[examples]"
```

## Examples

### 1. Basic Operations (`basic_operations.py`)

Demonstrates core HDC operations with performance timing and detailed explanations.

**Concepts:**
- Binding and unbinding (with commutativity check)
- Bundling and capacity limits
- Sequence encoding with permutation
- BSC vs MAP model comparison

**Run:**
```bash
python examples/basic_operations.py
```

**New features:**
- Performance benchmarks with timing
- Proper PRNG key splitting (no reuse)
- Expected value ranges documented
- Capacity testing for bundling operations

### 2. Kanerva's Dollar of Mexico (`kanerva_example.py`)

Classic HDC demonstration of structured knowledge and analogical reasoning.

**Concepts:**
- Role-filler binding
- Structural mapping
- Analogical queries
- Similarity-based memory search

**Run:**
```bash
python examples/kanerva_example.py
```

**Reference:**
Kanerva (2010). "What's the Dollar of Mexico?" AAAI Fall Symposium.
Paper: https://redwood.berkeley.edu/wp-content/uploads/2020/05/kanerva2010what.pdf

**New features:**
- Improved key management
- Enhanced explanations of role-filler binding
- Clearer analogical reasoning process

### 3. Classification (`classification_simple.py`)

End-to-end classification pipeline with performance analysis.

**Concepts:**
- Feature encoding (RandomEncoder)
- One-shot learning (CentroidClassifier)
- Confusion matrix analysis
- Complexity characteristics

**Run:**
```bash
python examples/classification_simple.py
```

**New features:**
- Training time measurement
- Returns metrics dictionary
- O(n) training complexity demonstration
- Detailed performance breakdown

## Code Quality

All examples include:
- Proper PRNG key splitting
- Performance timing with `time.perf_counter()`
- Expected output ranges
- Comprehensive docstrings
- Educational comments
- Syntax validation

## Educational Notes

**HDC Principles:**
- High dimensionality (10k+ dims) enables quasi-orthogonality
- Binding creates dissimilar vectors (encoding)
- Bundling creates similar vectors (aggregation)
- Similarity measures relatedness

**Model Selection:**
- BSC: Memory-efficient, binary operations
- MAP: Gradient-friendly, smooth optimization
- HRR: Circular convolution, theoretically grounded
- FHRR: Complex-valued, FFT-based

## Troubleshooting

**Import errors:**
```bash
pip install -e .
```

**JAX not found:**
```bash
pip install jax jaxlib
```

**Slow first run:**
JAX compilation overhead (JIT). Subsequent runs are faster.
