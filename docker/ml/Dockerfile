# Dockerfile for Machine Learning & Interpretability Container
# Python 3.14 image with SWI-Prolog, uv, PyTorch, Lightning, MLflow, and Marimo

ARG PYTHON_VERSION=3.14
ARG UV_VERSION=0.13.1
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv_bin
FROM python:${PYTHON_VERSION}-slim

# Use /opt/venv so bind-mounting .:/workspace does not shadow the container's Linux venv
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT="/opt/venv" \
    PATH="/opt/venv/bin:$PATH" \
    MLFLOW_TRACKING_URI="http://mlflow:5050"

# Install system dependencies including SWI-Prolog first (isolated from uv/python changes)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    ca-certificates \
    swi-prolog-nox \
    && rm -rf /var/lib/apt/lists/*

# Install pinned uv package manager
COPY --from=uv_bin /uv /uvx /bin/

WORKDIR /workspace

# 1. Copy only dependency manifests and install third-party dependencies first
COPY pyproject.toml uv.lock README.md /workspace/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --extra mlflow --group dev --group notebooks

# 2. Copy project source and install local package without re-installing dependencies
COPY src /workspace/src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra mlflow --group dev --group notebooks

# Ensure runtime directories exist
RUN mkdir -p /workspace/.cache \
             /workspace/datasets \
             /workspace/models \
             /workspace/data \
             /workspace/runs \
             /artifacts

EXPOSE 2718

# Health check - verify Marimo server and SWI-Prolog are responsive
HEALTHCHECK --interval=10s --timeout=15s --start-period=10s --retries=5 \
    CMD curl -sf http://localhost:${MARIMO_PORT:-2718} > /dev/null && swipl --version > /dev/null || exit 1

CMD ["sh", "-c", "marimo run ${ACTIVE_NOTEBOOK:-notebooks/logistic_demo.py} --host 0.0.0.0 --port ${MARIMO_PORT:-2718} --no-token"]
