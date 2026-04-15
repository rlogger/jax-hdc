{ pkgs ? import <nixpkgs> { }
, python ? pkgs.python311
, withDev ? false
, withExamples ? false
, withDocs ? false
}:

let
  pythonPackages = python.pkgs;

  coreDeps = with pythonPackages; [
    jax
    jaxlib
  ];

  devDeps = with pythonPackages; [
    pytest
    pytest-cov
    ruff
    mypy
  ];

  examplesDeps = with pythonPackages; [
    matplotlib
    scikit-learn
    tqdm
  ];

  docsDeps = with pythonPackages; [
    sphinx
    sphinx-rtd-theme
    myst-parser
  ];

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

  doCheck = !withDev;

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
}
