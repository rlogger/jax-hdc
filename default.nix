# Default.nix for traditional Nix users (non-flakes)
# This provides compatibility for users not using flakes
{ pkgs ? import <nixpkgs> { }
, python ? pkgs.python311
, withDev ? false
, withExamples ? false
, withDocs ? false
}:

let
  pythonPackages = python.pkgs;

  # Core runtime dependencies
  coreDeps = with pythonPackages; [
    jax
    jaxlib
    numpy
    optax
  ];

  # Development dependencies
  devDeps = with pythonPackages; [
    pytest
    pytest-cov
    black
    flake8
    mypy
    isort
  ];

  # Examples dependencies
  examplesDeps = with pythonPackages; [
    matplotlib
    scikit-learn
    tqdm
  ];

  # Documentation dependencies
  docsDeps = with pythonPackages; [
    sphinx
    sphinx-rtd-theme
    myst-parser
  ];

  # Determine which dependencies to include
  allDeps = coreDeps
    ++ (if withDev then devDeps else [ ])
    ++ (if withExamples then examplesDeps else [ ])
    ++ (if withDocs then docsDeps else [ ]);

in
pythonPackages.buildPythonPackage rec {
  pname = "jax-hdc";
  version = "0.1.0";
  format = "pyproject";

  src = ./.;

  nativeBuildInputs = with pythonPackages; [
    setuptools
    wheel
  ];

  propagatedBuildInputs = allDeps;

  nativeCheckInputs = with pythonPackages; [
    pytestCheckHook
    pytest-cov
  ];

  # Run tests unless building with dev mode (to avoid duplication)
  doCheck = !withDev;

  checkPhase = ''
    runHook preCheck
    pytest tests/ -v
    runHook postCheck
  '';

  pythonImportsCheck = [ "jax_hdc" ];

  meta = with pkgs.lib; {
    description = "Hyperdimensional Computing with JAX";
    longDescription = ''
      JAX-HDC is a library for Hyperdimensional Computing (HDC) / Vector Symbolic
      Architectures (VSA) built on JAX. It provides high-performance implementations
      of various HDC operations with hardware acceleration support.

      Features:
      - Multiple VSA models (BSC, MAP, HRR, FHRR)
      - Feature encoders (Random, Level, Projection)
      - Classification models (Centroid, Adaptive)
      - Full JAX integration with JIT compilation and hardware acceleration
    '';
    homepage = "https://github.com/rlogger/jax-hdc";
    license = licenses.mit;
    maintainers = [ ];
    platforms = platforms.unix;
  };
}
