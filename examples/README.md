# JAX-HDC Examples

This directory contains examples demonstrating how to use JAX-HDC for various tasks.

## Examples

- `basic_operations.py` - Demonstrates core HDC operations (binding, bundling, similarity)
- `classification_simple.py` - Simple classification example with synthetic data
- `kanerva_example.py` - Classic Kanerva "Dollar of Mexico" example

## Running Examples

Install JAX-HDC with example dependencies:

```bash
pip install -e ".[examples]"
```

Then run any example:

```bash
python examples/basic_operations.py
python examples/classification_simple.py
python examples/kanerva_example.py
```

## Requirements

Examples require additional dependencies listed in `pyproject.toml` under `[project.optional-dependencies.examples]`.
