# JAX-HDC Performance Benchmarks

Reproducible performance benchmarks for core HDC operations.

## Methodology

- **Dimensions**: 10,000 (standard HDC hypervector size)
- **Warmup**: 10 iterations (covers JIT compilation)
- **Trials**: 100 iterations per benchmark
- **Report**: Mean ± Std, Min, Max (milliseconds)

## Running Benchmarks

```bash
# Run benchmark script
python benchmarks/benchmark_operations.py

# From project root with venv
source .venv/bin/activate
python benchmarks/benchmark_operations.py
```

## Benchmarked Operations

| Operation | Description |
|-----------|-------------|
| BSC bind | XOR binding of binary hypervectors |
| BSC bundle | Majority-rule bundling |
| BSC similarity | Hamming similarity |
| MAP bind | Element-wise multiplication |
| MAP bundle | Normalized sum |
| MAP similarity | Cosine similarity |
| RandomEncoder | Encode 100 samples × 20 features |

## Reproducibility

For comparable results:
1. Use the same hardware (CPU/GPU)
2. Close other heavy processes
3. Run multiple times and report mean ± std

## Comparison with NumPy/TorchHD

To compare with NumPy or TorchHD, implement equivalent operations and run on the same hardware with the same dimensions. The benchmark script provides baseline JAX-HDC timings.
