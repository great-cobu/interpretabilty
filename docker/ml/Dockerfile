# Dockerfile for Machine Learning development
# Python image with SWI-Prolog, uv, ML libraries, PyTorch, and HuggingFace support

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

# Install uv package manager from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set environment variables for uv, Python, and HuggingFace
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    HF_HOME="/workspace/.cache/huggingface" \
    TRANSFORMERS_CACHE="/workspace/.cache/huggingface/transformers" \
    HF_DATASETS_CACHE="/workspace/.cache/huggingface/datasets" \
    MLFLOW_TRACKING_URI="http://mlflow:5000"

# Install system dependencies including SWI-Prolog
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    ca-certificates \
    swi-prolog \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy dependency specification first for Docker layer caching
COPY requirements.txt /workspace/requirements.txt

# Install all Python dependencies from requirements.txt using uv
RUN uv pip install --system --no-cache-dir -r /workspace/requirements.txt

# Copy the entire project directory into /workspace
COPY . /workspace

# Install the project itself in editable mode
RUN uv pip install --system --no-deps -e /workspace

# Create directory structure for datasets, models, and HuggingFace cache
RUN mkdir -p /workspace/.cache/huggingface/transformers \
             /workspace/.cache/huggingface/datasets \
             /workspace/datasets \
             /workspace/models

# Health check - verify key ML libraries import and SWI-Prolog runs
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import numpy, pandas, sklearn, torch, transformers, mlflow; print('OK')" && swipl --version || exit 1

# Volume for code mounting (optional override in compose)
VOLUME ["/workspace"]

# Default command - interactive bash shell
CMD ["bash"]
