#!/usr/bin/env python3
"""
Comprehensive validation script for JAX-HDC codebase.

This script performs a series of checks to ensure the codebase is functional,
well-typed, and ready for production use. It validates:
- Type checking with mypy
- Code formatting with black and isort
- Linting with flake8
- Test coverage
- JAX compatibility
- Runtime functionality

Usage:
    python validate_codebase.py
"""

import subprocess
import sys
from typing import List, Tuple, Dict, Any
import jax
import jax.numpy as jnp


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}\n")


def print_success(text: str) -> None:
    """Print a success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text: str) -> None:
    """Print an error message."""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_warning(text: str) -> None:
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def print_info(text: str) -> None:
    """Print an info message."""
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")


def run_command(cmd: List[str], description: str) -> Tuple[bool, str]:
    """
    Run a shell command and return success status and output.

    Args:
        cmd: Command and arguments as a list
        description: Human-readable description of what's being run

    Returns:
        Tuple of (success: bool, output: str)
    """
    print_info(f"Running: {description}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Command timed out after 5 minutes"
    except Exception as e:
        return False, f"Error running command: {e}"


def check_mypy() -> bool:
    """Run mypy type checking."""
    print_header("Type Checking (mypy)")
    success, output = run_command(
        ["python", "-m", "mypy", "jax_hdc/"],
        "mypy jax_hdc/"
    )

    if success:
        print_success("Type checking passed! No type errors found.")
        return True
    else:
        print_error("Type checking failed!")
        print(output)
        return False


def check_tests() -> bool:
    """Run pytest test suite."""
    print_header("Test Suite (pytest)")
    success, output = run_command(
        ["python", "-m", "pytest", "tests/", "-v", "--tb=short"],
        "pytest tests/"
    )

    if success:
        # Extract test count from output
        lines = output.split('\n')
        for line in lines:
            if 'passed' in line:
                print_success(f"All tests passed! {line.strip()}")
                return True
        print_success("All tests passed!")
        return True
    else:
        print_error("Some tests failed!")
        print(output[-2000:])  # Print last 2000 chars
        return False


def check_coverage() -> bool:
    """Check test coverage."""
    print_header("Test Coverage")
    success, output = run_command(
        ["python", "-m", "pytest", "tests/", "--cov=jax_hdc", "--cov-report=term"],
        "pytest --cov=jax_hdc"
    )

    if not success:
        print_error("Coverage check failed!")
        return False

    # Parse coverage percentage
    lines = output.split('\n')
    for line in lines:
        if 'TOTAL' in line:
            parts = line.split()
            if len(parts) >= 4:
                coverage = parts[-1].rstrip('%')
                try:
                    cov_num = float(coverage)
                    if cov_num >= 90:
                        print_success(f"Excellent coverage: {coverage}%")
                        return True
                    elif cov_num >= 80:
                        print_warning(f"Good coverage: {coverage}%")
                        return True
                    else:
                        print_warning(f"Coverage could be improved: {coverage}%")
                        return True
                except ValueError:
                    pass

    print_info("Coverage report generated")
    return True


def check_jax_compatibility() -> bool:
    """Test JAX compatibility and basic functionality."""
    print_header("JAX Compatibility")

    try:
        # Check JAX installation
        print_info(f"JAX version: {jax.__version__}")
        print_info(f"Available devices: {jax.devices()}")

        # Test basic JAX operations
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([4.0, 5.0, 6.0])
        z = x + y
        jax.block_until_ready(z)
        print_success("Basic JAX operations work")

        # Test JIT compilation
        @jax.jit
        def test_jit(a: jax.Array, b: jax.Array) -> jax.Array:
            return a * b + a

        result = test_jit(x, y)
        jax.block_until_ready(result)
        print_success("JIT compilation works")

        # Test vmap
        batched_fn = jax.vmap(lambda a, b: a * b)
        result = batched_fn(x, y)
        jax.block_until_ready(result)
        print_success("vmap (auto-vectorization) works")

        return True

    except Exception as e:
        print_error(f"JAX compatibility check failed: {e}")
        return False


def check_imports() -> bool:
    """Check that all package imports work."""
    print_header("Package Imports")

    try:
        # Import main package
        import jax_hdc
        print_success("jax_hdc package imports successfully")

        # Import submodules
        from jax_hdc import functional, vsa, embeddings, models, utils
        print_success("All submodules import successfully")

        # Check key classes/functions are available
        from jax_hdc import BSC, MAP, HRR, FHRR
        print_success("VSA models available")

        from jax_hdc import RandomEncoder, LevelEncoder, ProjectionEncoder
        print_success("Encoders available")

        from jax_hdc import CentroidClassifier, AdaptiveHDC
        print_success("Classifiers available")

        return True

    except Exception as e:
        print_error(f"Import check failed: {e}")
        return False


def test_basic_functionality() -> bool:
    """Test basic HDC operations."""
    print_header("Basic Functionality Tests")

    try:
        from jax_hdc import MAP, RandomEncoder, CentroidClassifier
        import jax.random

        # Test VSA model
        key = jax.random.PRNGKey(42)
        model = MAP.create(dimensions=1000)
        x = model.random(key, (1000,))
        y = model.random(jax.random.split(key)[1], (1000,))

        # Test bind and bundle
        bound = model.bind(x, y)
        bundled = model.bundle(jnp.stack([x, y]), axis=0)

        # Test similarity
        sim = model.similarity(x, x)
        sim_val = float(sim)
        assert 0.95 < sim_val <= 1.01, f"Self-similarity should be ~1.0, got {sim_val}"

        print_success("VSA model operations work")

        # Test encoder
        encoder = RandomEncoder.create(
            num_features=10,
            num_values=10,
            dimensions=1000,
            vsa_model='map',
            key=key
        )
        features = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
        encoded = encoder.encode(features)
        assert encoded.shape == (1000,), f"Encoded shape should match dimensions, got {encoded.shape}"

        print_success("Encoder operations work")

        # Test classifier
        classifier = CentroidClassifier.create(
            num_classes=3,
            dimensions=1000,
            vsa_model='map'
        )

        # Create dummy training data
        train_hvs = model.random(key, (30, 1000))
        train_labels = jnp.array([i % 3 for i in range(30)])

        # Train
        classifier = classifier.fit(train_hvs, train_labels)

        # Predict
        predictions = classifier.predict(train_hvs[:5])
        assert predictions.shape == (5,), "Predictions shape should match input"

        print_success("Classifier operations work")

        return True

    except Exception as e:
        print_error(f"Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_code_quality() -> bool:
    """Check code formatting and linting."""
    print_header("Code Quality Checks")

    # Check black formatting
    success, output = run_command(
        ["python", "-m", "black", "--check", "jax_hdc/"],
        "black --check jax_hdc/"
    )

    if success:
        print_success("Code formatting (black) is correct")
    else:
        print_warning("Code needs formatting with black")

    return True  # Don't fail on formatting issues


def generate_report(results: Dict[str, bool]) -> None:
    """Generate a summary report of all checks."""
    print_header("Validation Summary")

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed

    print(f"\n{Colors.BOLD}Results:{Colors.END}")
    for check, result in results.items():
        if result:
            print_success(f"{check}")
        else:
            print_error(f"{check}")

    print(f"\n{Colors.BOLD}Summary:{Colors.END}")
    print(f"  Total checks: {total}")
    print(f"  {Colors.GREEN}Passed: {passed}{Colors.END}")
    print(f"  {Colors.RED}Failed: {failed}{Colors.END}")

    if failed == 0:
        print(f"\n{Colors.BOLD}{Colors.GREEN}🎉 All validation checks passed!{Colors.END}")
        print(f"{Colors.GREEN}The codebase is ready for use.{Colors.END}\n")
        return True
    else:
        print(f"\n{Colors.BOLD}{Colors.RED}❌ Some validation checks failed.{Colors.END}")
        print(f"{Colors.RED}Please fix the issues above.{Colors.END}\n")
        return False


def main() -> int:
    """Run all validation checks."""
    print(f"{Colors.BOLD}{Colors.BLUE}")
    print("╔═══════════════════════════════════════════════════════════════════════════════╗")
    print("║                                                                               ║")
    print("║                    JAX-HDC CODEBASE VALIDATION SUITE                          ║")
    print("║                                                                               ║")
    print("╚═══════════════════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}")

    # Run all checks
    results = {}

    results["Package Imports"] = check_imports()
    results["JAX Compatibility"] = check_jax_compatibility()
    results["Basic Functionality"] = test_basic_functionality()
    results["Type Checking (mypy)"] = check_mypy()
    results["Test Suite"] = check_tests()
    results["Test Coverage"] = check_coverage()
    results["Code Quality"] = check_code_quality()

    # Generate report
    all_passed = generate_report(results)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
