# JAX-HDC Performance Benchmarks

Reproducible performance benchmarks for core HDC operations and comparison with [TorchHD](https://github.com/hyperdimensional-computing/torchhd).

## Methodology

- **Dimensions**: 10,000 (standard HDC hypervector size)
- **Warmup**: 20 iterations (covers JIT compilation)
- **Trials**: 200 per operation
- **Device**: CPU (both libraries) for fair comparison
- **Report**: Mean ± Std (milliseconds)

## Running Benchmarks

### JAX-HDC only

```bash
python benchmarks/benchmark_operations.py
```

### JAX-HDC vs TorchHD comparison

```bash
pip install -e ".[benchmark]"
python benchmarks/benchmark_compare.py
```

Results are printed to stdout and saved to `benchmark_results.json`.

## Benchmarked Operations

| Operation | Description |
|-----------|-------------|
| MAP bind (2 HVs) | Element-wise multiply of two MAP hypervectors |
| MAP bundle (10 HVs) | Normalized sum of 10 MAP hypervectors |
| Cosine similarity | Cosine similarity between two vectors |
| RandomEncoder (100×20) | Encode 100 samples × 20 discrete features |

## Reproducibility

For comparable results:
1. Use the same hardware (CPU/GPU)
2. Close other heavy processes
3. Run multiple times; results vary by machine

## Typical Results (CPU)

| Operation | JAX-HDC | TorchHD | Speedup |
|-----------|---------|---------|---------|
| MAP bind | ~0.01 ms | ~0.01 ms | ~1× |
| MAP bundle | ~0.04 ms | ~0.03 ms | ~1× |
| Cosine similarity | ~0.02 ms | ~0.07 ms | **~3×** |
| RandomEncoder | ~1.0 ms | ~1.0 ms | ~1× |

JAX-HDC shows ~3× speedup on similarity; other ops are comparable. GPU/TPU may yield larger gains for JAX.
