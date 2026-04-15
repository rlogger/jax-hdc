# Contributing to JAX-HDC

Thank you for your interest in contributing to JAX-HDC.

## Development Setup

```bash
git clone https://github.com/rlogger/jax-hdc.git
cd jax-hdc
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Code Style

We use `ruff` for linting and formatting (line length: 100, target: Python 3.9+):

```bash
ruff check jax_hdc/ tests/
ruff format jax_hdc/ tests/
mypy jax_hdc/
```

### Docstrings

Use Google-style docstrings:

```python
def function_name(arg1: type, arg2: type) -> return_type:
    """Brief description.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value
    """
```

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=jax_hdc --cov-report=term-missing
```

Add tests for all new functionality in the `tests/` directory.

## Functional Programming Principles

JAX-HDC follows functional programming principles:

1. **Pure functions**: No side effects
2. **Immutability**: Use `.at[]` syntax for updates, return new instances
3. **Explicit state**: Pass state as arguments
4. **JIT-compatible**: Avoid Python control flow in JIT functions

## Pull Request Process

1. Create a feature branch
2. Add tests for new functionality
3. Ensure `ruff check`, `ruff format --check`, and `pytest` all pass
4. Submit a PR with a clear description

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
