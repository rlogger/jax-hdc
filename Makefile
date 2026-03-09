.PHONY: help install install-dev install-all test lint format typecheck docs docs-serve build publish publish-test clean

PYTHON ?= python3
PIP    ?= $(PYTHON) -m pip

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Install the package
	$(PIP) install -e .

install-dev: ## Install with dev dependencies
	$(PIP) install -e ".[dev]"

install-all: ## Install with all optional dependencies
	$(PIP) install -e ".[dev,docs,examples,benchmark]"

test: ## Run tests with coverage
	$(PYTHON) -m pytest tests/ -v --cov=jax_hdc --cov-report=term-missing -k "not benchmark"

test-all: ## Run all tests including benchmarks
	$(PYTHON) -m pytest tests/ -v --cov=jax_hdc --cov-report=term-missing

lint: ## Run linter
	ruff check jax_hdc/ tests/ examples/ benchmarks/

format: ## Auto-format code
	ruff format jax_hdc/ tests/ examples/ benchmarks/
	ruff check --fix jax_hdc/ tests/ examples/ benchmarks/

typecheck: ## Run type checker
	mypy jax_hdc/ --ignore-missing-imports

docs: ## Build Sphinx documentation
	$(PYTHON) -m sphinx -b html docs/ docs/_build/html

docs-serve: docs ## Build and open docs in browser
	$(PYTHON) -m webbrowser docs/_build/html/index.html

build: clean ## Build sdist and wheel
	$(PIP) install --upgrade build
	$(PYTHON) -m build

publish-test: build ## Publish to TestPyPI
	$(PIP) install --upgrade twine
	$(PYTHON) -m twine upload --repository testpypi dist/*

publish: build ## Publish to PyPI
	$(PIP) install --upgrade twine
	$(PYTHON) -m twine upload dist/*

clean: ## Remove build artifacts
	rm -rf dist/ build/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
