{
  description = "JAX-HDC: Hyperdimensional Computing with JAX";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python311;
        pythonPackages = python.pkgs;

        # Main package derivation
        jax-hdc = pythonPackages.buildPythonPackage {
          pname = "jax-hdc";
          version = "0.1.0";
          format = "pyproject";

          src = ./.;

          nativeBuildInputs = with pythonPackages; [
            setuptools
            wheel
          ];

          propagatedBuildInputs = with pythonPackages; [
            jax
            jaxlib
            numpy
            optax
          ];

          # Development and testing dependencies
          nativeCheckInputs = with pythonPackages; [
            pytestCheckHook
            pytest-cov
          ];

          # Run tests during build
          checkPhase = ''
            runHook preCheck
            pytest tests/ -v
            runHook postCheck
          '';

          pythonImportsCheck = [ "jax_hdc" ];

          meta = with pkgs.lib; {
            description = "Hyperdimensional Computing with JAX";
            homepage = "https://github.com/rlogger/jax-hdc";
            license = licenses.mit;
            maintainers = [ ];
            platforms = platforms.unix;
          };
        };

        # Development environment with all optional dependencies
        devEnv = pythonPackages.buildPythonPackage {
          pname = "jax-hdc-dev";
          version = "0.1.0";
          format = "pyproject";

          src = ./.;

          nativeBuildInputs = with pythonPackages; [
            setuptools
            wheel
          ];

          propagatedBuildInputs = with pythonPackages; [
            # Core dependencies
            jax
            jaxlib
            numpy
            optax

            # Dev dependencies
            pytest
            pytest-cov
            black
            flake8
            mypy
            isort

            # Examples dependencies
            matplotlib
            scikit-learn
            tqdm

            # Docs dependencies
            sphinx
            sphinx-rtd-theme
            myst-parser
          ];

          dontCheck = true;

          meta = with pkgs.lib; {
            description = "JAX-HDC development environment";
          };
        };

      in
      {
        packages = {
          default = jax-hdc;
          jax-hdc = jax-hdc;
          dev = devEnv;
        };

        # Development shell
        devShells.default = pkgs.mkShell {
          buildInputs = [
            devEnv
            python
          ];

          shellHook = ''
            echo "JAX-HDC development environment"
            echo "Python version: $(python --version)"
            echo ""
            echo "Available commands:"
            echo "  pytest tests/          - Run tests"
            echo "  black jax_hdc/         - Format code"
            echo "  flake8 jax_hdc/        - Lint code"
            echo "  mypy jax_hdc/          - Type check"
            echo "  python examples/*.py   - Run examples"
            echo ""
            echo "To install in editable mode:"
            echo "  pip install -e ."
          '';
        };

        # Apps for running examples
        apps = {
          basic-operations = {
            type = "app";
            program = "${pkgs.writeShellScript "basic-operations" ''
              ${python}/bin/python ${./examples/basic_operations.py}
            ''}";
          };

          kanerva-example = {
            type = "app";
            program = "${pkgs.writeShellScript "kanerva-example" ''
              ${python}/bin/python ${./examples/kanerva_example.py}
            ''}";
          };

          classification-simple = {
            type = "app";
            program = "${pkgs.writeShellScript "classification-simple" ''
              ${python}/bin/python ${./examples/classification_simple.py}
            ''}";
          };
        };
      }
    );
}
