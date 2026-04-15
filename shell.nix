{ pkgs ? import <nixpkgs> { }
, python ? pkgs.python311
}:

let
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
    pkgs.git
    pkgs.pre-commit
  ];

  shellHook = ''
    echo "JAX-HDC Development Environment"
    echo "Python: $(python --version)"
    echo ""
    echo "Commands:"
    echo "  pytest tests/                 - Run tests"
    echo "  pytest tests/ --cov=jax_hdc   - Tests with coverage"
    echo "  ruff check jax_hdc/ tests/    - Lint"
    echo "  ruff format jax_hdc/ tests/   - Format"
    echo "  mypy jax_hdc/                 - Type check"
    echo ""
    echo "Examples:"
    echo "  python examples/basic_operations.py"
    echo "  python examples/kanerva_example.py"
    echo "  python examples/classification_simple.py"
  '';

  JAX_PLATFORM_NAME = "cpu";
  JAX_ENABLE_X64 = "True";
}
