# shell.nix for development environment
# Usage: nix-shell
{ pkgs ? import <nixpkgs> { }
, python ? pkgs.python311
}:

let
  pythonPackages = python.pkgs;

  # Import the package with all dev dependencies
  jax-hdc-dev = import ./default.nix {
    inherit pkgs python;
    withDev = true;
    withExamples = true;
    withDocs = true;
  };

in
pkgs.mkShell {
  name = "jax-hdc-dev-shell";

  buildInputs = [
    python
    jax-hdc-dev

    # Additional development tools
    pkgs.git
    pkgs.pre-commit
  ];

  shellHook = ''
    echo "========================================="
    echo "  JAX-HDC Development Environment"
    echo "========================================="
    echo ""
    echo "Python: $(python --version)"
    echo "JAX: $(python -c 'import jax; print(jax.__version__)' 2>/dev/null || echo 'not installed')"
    echo ""
    echo "Available commands:"
    echo "  pytest tests/                 - Run tests"
    echo "  pytest tests/ --cov=jax_hdc   - Run tests with coverage"
    echo "  black jax_hdc/ tests/         - Format code"
    echo "  flake8 jax_hdc/ tests/        - Lint code"
    echo "  isort jax_hdc/ tests/         - Sort imports"
    echo "  mypy jax_hdc/                 - Type check"
    echo ""
    echo "Examples:"
    echo "  python examples/basic_operations.py"
    echo "  python examples/kanerva_example.py"
    echo "  python examples/classification_simple.py"
    echo ""
    echo "Documentation:"
    echo "  cd docs && make html          - Build HTML docs"
    echo "  cd docs && make clean         - Clean docs"
    echo ""
    echo "Install in editable mode:"
    echo "  pip install -e .              - Editable install (basic)"
    echo "  pip install -e '.[dev]'       - With dev dependencies"
    echo "  pip install -e '.[examples]'  - With example dependencies"
    echo "  pip install -e '.[docs]'      - With doc dependencies"
    echo ""
    echo "========================================="

    # Set up Python path to allow running from source
    export PYTHONPATH="${pythonPackages.jax}/lib/${python.libPrefix}/site-packages:$PYTHONPATH"

    # Helpful aliases
    alias test='pytest tests/ -v'
    alias test-cov='pytest tests/ -v --cov=jax_hdc --cov-report=term-missing'
    alias format='black jax_hdc/ tests/ examples/ && isort jax_hdc/ tests/ examples/'
    alias lint='flake8 jax_hdc/ tests/ examples/'
    alias typecheck='mypy jax_hdc/'
  '';

  # Environment variables
  PYTHONBREAKPOINT = "ipdb.set_trace";

  # JAX configuration
  JAX_PLATFORM_NAME = "cpu";  # Change to "gpu" or "tpu" if available
  JAX_ENABLE_X64 = "True";     # Enable 64-bit precision
}
