# Contributing to JAX-HDC

Thank you for your interest in contributing to JAX-HDC! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and constructive in all interactions. We aim to maintain a welcoming and inclusive community.

## Getting Started

### Setting Up Development Environment

1. Fork and clone the repository:

```bash
git clone https://github.com/rlogger/jax-hdc.git
cd jax-hdc
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:

```bash
pip install -e ".[dev]"
```

### Development Workflow

1. Create a new branch for your feature:

```bash
git checkout -b feature/your-feature-name
```

2. Make your changes
3. Run tests to ensure nothing breaks:

```bash
pytest tests/ -v
```

4. Format your code:

```bash
black jax_hdc/
isort jax_hdc/
```

5. Commit your changes:

```bash
git add .
git commit -m "Add: brief description of changes"
```

6. Push to your fork and create a Pull Request

## Contribution Guidelines

### Code Style

- Follow PEP 8 guidelines
- Use `black` for code formatting (line length: 100)
- Use `isort` for import sorting
- Add type hints where appropriate
- Write docstrings for all public functions and classes

### Docstring Format

Use Google-style docstrings:

```python
def function_name(arg1: type, arg2: type) -> return_type:
    """Brief description of function.

    Longer description if needed.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value

    Example:
        >>> result = function_name(1, 2)
        >>> print(result)
        3
    """
    pass
```

### Writing Tests

- Add tests for all new functionality
- Tests should be in the `tests/` directory
- Use descriptive test names: `test_bind_bsc_commutativity`
- Aim for high test coverage (>80%)

Example test:

```python
def test_new_feature():
    """Test that new feature works correctly."""
    # Setup
    model = MAP.create(dimensions=10000)
    key = jax.random.PRNGKey(42)

    # Exercise
    result = model.new_feature(key)

    # Verify
    assert result.shape == (10000,)
    assert jnp.all(jnp.isfinite(result))
```

### Functional Programming Principles

JAX-HDC follows functional programming principles:

1. **Pure Functions**: No side effects
2. **Immutability**: Use `.at[]` syntax for updates
3. **Explicit State**: Pass state as arguments
4. **JIT-Compatible**: Avoid Python control flow in JIT functions

Example:

```python
@jax.jit
def good_function(x: jax.Array, y: jax.Array) -> jax.Array:
    """Pure function, no side effects."""
    result = x + y
    return result / jnp.linalg.norm(result)

# Avoid global state
bad_counter = 0  # Don't do this

def bad_function(x):
    global bad_counter  # Don't do this
    bad_counter += 1
    return x + bad_counter
```

## Areas for Contribution

### High Priority

1. **Additional VSA Models**: B-SBC, CGR, MCR, VTB
2. **Memory Modules**: Sparse Distributed Memory, Hopfield networks
3. **Performance Optimizations**: Custom XLA kernels, better fusion
4. **Documentation**: Examples, tutorials, API docs

### Medium Priority

1. **Encoders**: Graph encoders, kernel approximation
2. **Datasets**: Benchmark dataset loaders
3. **Learning Algorithms**: Gradient-based methods, online learning
4. **Visualization Tools**: Plotting utilities

### Low Priority

1. **Utilities**: Additional helper functions
2. **Examples**: More application examples
3. **Benchmarks**: Performance comparison scripts

## Implementing New VSA Models

To add a new VSA model:

1. Create a new class inheriting from `VSAModel` in `vsa.py`:

```python
@jax.tree_util.register_dataclass
@dataclass
class NewModel(VSAModel):
    """Description of new model."""

    name: str = field(default="newmodel", metadata=dict(static=True))

    @staticmethod
    def create(dimensions: int = 10000) -> "NewModel":
        """Create a NewModel instance."""
        return NewModel(name="newmodel", dimensions=dimensions)

    @jax.jit
    def bind(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Implement binding operation."""
        pass

    @jax.jit
    def bundle(self, vectors: jax.Array, axis: int = 0) -> jax.Array:
        """Implement bundling operation."""
        pass

    # ... implement other methods
```

2. Add tests in `tests/test_vsa.py`:

```python
class TestNewModel:
    """Test NewModel operations."""

    def test_creation(self):
        model = NewModel.create(dimensions=10000)
        assert model.dimensions == 10000

    # ... more tests
```

3. Update documentation and examples

## Implementing New Encoders

To add a new encoder:

1. Create a new class in `embeddings.py`:

```python
@jax.tree_util.register_dataclass
@dataclass
class NewEncoder:
    """Description of new encoder."""

    # Data fields
    data: jax.Array

    # Metadata fields
    dimensions: int = field(metadata=dict(static=True))

    @staticmethod
    def create(...) -> "NewEncoder":
        """Factory method."""
        pass

    @jax.jit
    def encode(self, x: jax.Array) -> jax.Array:
        """Encode input as hypervector."""
        pass
```

2. Add comprehensive tests
3. Add example usage in documentation

## Pull Request Process

1. Update the README.md if needed
2. Add tests for new functionality
3. Ensure all tests pass
4. Update documentation
5. Create a Pull Request with:
   - Clear title describing the change
   - Description of what changed and why
   - Link to any related issues
   - Screenshots/examples if applicable

### PR Checklist

- [ ] Code follows the style guidelines
- [ ] Tests added and passing
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] No merge conflicts

## Questions?

If you have questions, please:

1. Check existing issues and documentation
2. Open a new issue with your question
3. Tag it with the "question" label

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
