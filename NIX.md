# Nix Packaging for JAX-HDC

This document provides detailed information about using JAX-HDC with Nix/NixOS.

## Overview

JAX-HDC provides comprehensive Nix support through:
- **flake.nix**: Modern Nix flakes for reproducible builds
- **default.nix**: Traditional Nix expression
- **shell.nix**: Development environment
- **.envrc**: direnv integration for automatic environment loading

## Quick Start

### Using Nix Flakes (Recommended)

```bash
# Enter development shell
nix develop

# Build the package
nix build

# Run the package (examples)
nix run .#basic-operations
nix run .#kanerva-example
nix run .#classification-simple

# Check flake
nix flake check
```

### Using Traditional Nix

```bash
# Enter development shell
nix-shell

# Build the package
nix-build

# Build with specific options
nix-build --arg withDev true --arg withExamples true
```

## Package Outputs

The flake provides several outputs:

### Packages

- `packages.default`: The JAX-HDC library with core dependencies
- `packages.jax-hdc`: Same as default
- `packages.dev`: Development environment with all optional dependencies

```bash
# Build specific package
nix build .#jax-hdc
nix build .#dev
```

### Development Shells

The `devShells.default` provides a complete development environment including:

**Core Dependencies:**
- jax >= 0.4.20
- jaxlib >= 0.4.20
- numpy >= 1.22.0
- optax >= 0.1.7

**Development Tools:**
- pytest, pytest-cov
- black, flake8, isort, mypy

**Example Dependencies:**
- matplotlib, scikit-learn, tqdm

**Documentation Tools:**
- sphinx, sphinx-rtd-theme, myst-parser

```bash
# Enter development shell
nix develop

# Or with specific Python version (if customized)
nix develop --override-input python python311
```

### Apps

The flake exposes example scripts as apps:

```bash
# Run basic operations example
nix run .#basic-operations

# Run Kanerva's "Dollar of Mexico" example
nix run .#kanerva-example

# Run classification example
nix run .#classification-simple
```

## Customization

### Using Different Python Versions

By default, the package uses Python 3.11. To use a different version:

```nix
# In your own flake or nix expression
{
  inputs.jax-hdc.url = "github:rlogger/jax-hdc";

  outputs = { self, nixpkgs, jax-hdc }: {
    # Override Python version
    packages.x86_64-linux.default = jax-hdc.packages.x86_64-linux.default.override {
      python = nixpkgs.legacyPackages.x86_64-linux.python312;
    };
  };
}
```

### Building with Optional Dependencies

Using `default.nix`:

```bash
# Build with development dependencies
nix-build --arg withDev true

# Build with examples dependencies
nix-build --arg withExamples true

# Build with documentation dependencies
nix-build --arg withDocs true

# Build with all dependencies
nix-build --arg withDev true --arg withExamples true --arg withDocs true
```

### Using in Your Own Project

#### As a Flake Input

```nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    jax-hdc.url = "github:rlogger/jax-hdc";
  };

  outputs = { self, nixpkgs, jax-hdc }: {
    # Use in your development shell
    devShells.x86_64-linux.default = nixpkgs.legacyPackages.x86_64-linux.mkShell {
      buildInputs = [
        jax-hdc.packages.x86_64-linux.default
      ];
    };
  };
}
```

#### As a Traditional Package

```nix
# In your Nix expression
let
  pkgs = import <nixpkgs> {};
  jax-hdc = pkgs.callPackage /path/to/jax-hdc {};
in
  pkgs.mkShell {
    buildInputs = [ jax-hdc ];
  }
```

## Development Workflow

### With direnv

For automatic environment loading, install [direnv](https://direnv.net/):

```bash
# Install direnv (on NixOS)
nix-env -iA nixos.direnv

# Or add to configuration.nix
environment.systemPackages = [ pkgs.direnv ];
programs.direnv.enable = true;

# In the project directory
direnv allow

# Environment loads automatically on cd
cd jax-hdc  # Nix shell activates automatically
```

The `.envrc` file is already configured to use flakes by default.

### Development Commands

Once in the development shell (`nix develop` or `nix-shell`):

```bash
# Run tests
pytest tests/ -v
pytest tests/ --cov=jax_hdc --cov-report=term-missing

# Code formatting
black jax_hdc/ tests/ examples/
isort jax_hdc/ tests/ examples/

# Linting
flake8 jax_hdc/ tests/ examples/

# Type checking
mypy jax_hdc/

# Build documentation
cd docs && make html

# Run examples
python examples/basic_operations.py
python examples/kanerva_example.py
python examples/classification_simple.py
```

The shell provides helpful aliases:
- `test`: Run pytest
- `test-cov`: Run pytest with coverage
- `format`: Run black and isort
- `lint`: Run flake8
- `typecheck`: Run mypy

### Editable Installation

For an editable installation (useful when developing):

```bash
# In nix-shell or nix develop
pip install -e .
pip install -e '.[dev]'
pip install -e '.[examples]'
```

## Hardware Acceleration

### GPU Support

JAX with GPU support requires CUDA. On NixOS:

```nix
# Override jaxlib to use CUDA version
{
  packages.default = jax-hdc.packages.x86_64-linux.default.overridePythonAttrs (old: {
    propagatedBuildInputs = old.propagatedBuildInputs ++ [
      pythonPackages.jax-cuda
      pythonPackages.jaxlib-cuda
    ];
  });
}
```

Or set environment variable in `shell.nix`:

```nix
# In shellHook
export JAX_PLATFORM_NAME="gpu"
```

### TPU Support

For TPU support, you'll need `jaxlib` built with TPU support. This is typically handled by Google Cloud's environment.

## CI/CD Integration

### GitHub Actions with Nix

```yaml
name: Nix Build

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: cachix/install-nix-action@v20
        with:
          extra_nix_config: |
            experimental-features = nix-command flakes

      - name: Build
        run: nix build

      - name: Run tests
        run: nix develop --command pytest tests/
```

### cachix for Binary Caching

For faster builds, consider using [cachix](https://cachix.org/):

```bash
# Install cachix
nix-env -iA cachix -f https://cachix.org/api/v1/install

# Use a cache (example)
cachix use your-cache-name

# Push to cache (if you maintain one)
nix build | cachix push your-cache-name
```

## Troubleshooting

### Flakes Not Enabled

If you get an error about flakes not being enabled:

```bash
# Enable flakes temporarily
nix --experimental-features 'nix-command flakes' develop

# Or enable permanently in ~/.config/nix/nix.conf
echo "experimental-features = nix-command flakes" >> ~/.config/nix/nix.conf
```

On NixOS, add to `/etc/nixos/configuration.nix`:

```nix
nix.settings.experimental-features = [ "nix-command" "flakes" ];
```

### JAX Installation Issues

If JAX fails to install or import:

```bash
# Check JAX version
nix develop --command python -c "import jax; print(jax.__version__)"

# Check available devices
nix develop --command python -c "import jax; print(jax.devices())"
```

### Build Failures

```bash
# Clean build artifacts
nix-collect-garbage
nix store gc

# Rebuild with verbose output
nix build --show-trace

# Check flake for issues
nix flake check --show-trace
```

## Platform Support

JAX-HDC Nix packaging is tested on:
- x86_64-linux
- aarch64-linux (ARM64)
- x86_64-darwin (macOS Intel)
- aarch64-darwin (macOS Apple Silicon)

The flake uses `flake-utils.lib.eachDefaultSystem` to support all platforms automatically.

## Resources

- [Nix Manual](https://nixos.org/manual/nix/stable/)
- [Nix Flakes](https://nixos.wiki/wiki/Flakes)
- [nixpkgs Python](https://nixos.org/manual/nixpkgs/stable/#python)
- [direnv](https://direnv.net/)
- [JAX Installation Guide](https://github.com/google/jax#installation)

## Contributing

When contributing Nix-related changes:

1. Test with both flakes and traditional Nix
2. Ensure all platforms build (use `nix flake check`)
3. Update this documentation if adding new features
4. Verify examples still run (`nix run .#basic-operations`)

## License

Same as JAX-HDC: MIT License
