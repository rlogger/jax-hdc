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
          ];

          nativeCheckInputs = with pythonPackages; [
            pytestCheckHook
            pytest-cov
          ];

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
            jax
            jaxlib
            pytest
            pytest-cov
            ruff
            mypy
            matplotlib
            scikit-learn
            tqdm
            sphinx
            sphinx-rtd-theme
            myst-parser
          ];

          dontCheck = true;

          meta.description = "JAX-HDC development environment";
        };

      in
      {
        packages = {
          default = jax-hdc;
          jax-hdc = jax-hdc;
          dev = devEnv;
        };

        devShells.default = pkgs.mkShell {
          buildInputs = [
            devEnv
            python
          ];

          shellHook = ''
            echo "JAX-HDC development environment"
            echo "Python: $(python --version)"
            echo ""
            echo "Commands:"
            echo "  pytest tests/          - Run tests"
            echo "  ruff check jax_hdc/    - Lint"
            echo "  ruff format jax_hdc/   - Format"
            echo "  mypy jax_hdc/          - Type check"
          '';
        };

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
