FROM python:3.12-slim AS base

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY jax_hdc/ jax_hdc/

RUN pip install --no-cache-dir .

FROM base AS dev

COPY tests/ tests/
COPY examples/ examples/
COPY benchmarks/ benchmarks/
COPY docs/ docs/
COPY Makefile ./

RUN pip install --no-cache-dir ".[dev,docs]"

CMD ["pytest", "tests/", "-v", "-k", "not benchmark"]

FROM base AS runtime

CMD ["python", "-c", "import jax_hdc; print(f'jax-hdc {jax_hdc.__version__} ready')"]
