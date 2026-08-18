# Dockerfile for MLFlow Tracking Server
# Python image with MLflow and database drivers

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install MLFlow and database/storage dependencies
RUN pip install --no-cache-dir \
    mlflow==2.11.1 \
    pymysql==1.1.0 \
    cryptography==42.0.5 \
    boto3==1.28.0 \
    google-cloud-storage==2.10.0 \
    psycopg2-binary==2.9.9

# Create MLFlow and artifact directories
RUN mkdir -p /mlflow /artifacts

WORKDIR /mlflow

# Expose default MLFlow port
EXPOSE 5000

# Default environment variables
ENV MLFLOW_PORT=5000
ENV MLFLOW_HOST=0.0.0.0

# Health check
HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${MLFLOW_PORT:-5000}/health || exit 1

# Default command (can be overridden by docker-compose)
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
