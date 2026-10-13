# Dockerfile for MLflow Tracking Server
# Python image with MLflow 3.x and MariaDB / S3 drivers

ARG PYTHON_VERSION=3.14
FROM python:${PYTHON_VERSION}-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install MLflow 3.x aligned with uv.lock plus database/storage drivers
RUN uv pip install --system --no-cache \
    "mlflow==3.15.2" \
    "pymysql>=1.1.0" \
    "cryptography>=42.0.5" \
    "boto3>=1.28.0"

RUN mkdir -p /mlflow /artifacts

WORKDIR /mlflow

EXPOSE 5000

ENV MLFLOW_PORT=5000 \
    MLFLOW_HOST=0.0.0.0

HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${MLFLOW_PORT:-5000}/health || exit 1

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
