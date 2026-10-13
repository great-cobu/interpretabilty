# Dockerfile for Machine Learning & Interpretability Container
# Python 3.14 image with SWI-Prolog, uv, PyTorch, Lightning, MLflow, and Marimo

ARG PYTHON_VERSION=3.14
FROM python:${PYTHON_VERSION}-slim

# Install uv package manager from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Use /opt/venv so bind-mounting .:/workspace does not shadow the container's Linux venv
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PROJECT_ENVIRONMENT="/opt/venv" \
    PATH="/opt/venv/bin:$PATH" \
    MLFLOW_TRACKING_URI="http://mlflow:5000"

# Install system dependencies including SWI-Prolog
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    ca-certificates \
    swi-prolog-nox \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy project dependency manifests first for Docker layer caching
COPY pyproject.toml uv.lock README.md /workspace/
COPY src /workspace/src

# Sync locked dependencies, MLflow extra, dev (pytest), and notebooks (marimo) into /opt/venv
RUN uv sync --locked --extra mlflow --group dev --group notebooks

# Copy remaining project files
COPY . /workspace

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
