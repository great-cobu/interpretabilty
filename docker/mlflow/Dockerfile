# Dockerfile for MLflow Tracking Server
# Python image with MLflow 3.x and MariaDB / S3 drivers

ARG PYTHON_VERSION=3.14
ARG UV_VERSION=0.13.1
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv_bin
FROM python:${PYTHON_VERSION}-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv_bin /uv /uvx /bin/

# Install MLflow 3.x aligned with uv.lock plus database/storage drivers
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
    "mlflow==3.15.2" \
    "pymysql>=1.1.0" \
    "cryptography>=42.0.5" \
    "boto3>=1.28.0"

RUN mkdir -p /mlflow /artifacts

WORKDIR /mlflow

EXPOSE 5050

ENV MLFLOW_PORT=5050 \
    MLFLOW_HOST=0.0.0.0

HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${MLFLOW_PORT:-5050}/health || exit 1

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5050"]
