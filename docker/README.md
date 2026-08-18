# Docker Stack Directory Structure

This directory houses the container build contexts (Dockerfiles, init scripts) and service environment files for the MLOps experiment tracking stack.

## Structure

- **`mariadb/`**: MariaDB metadata backend for MLflow
  - `Dockerfile`: MariaDB 11.4 LTS image
  - `init.sql`: Automatic database schema / collation init script
  - `.env`: Generated environment variables for database auth and ports
- **`mlflow/`**: MLflow Tracking Server
  - `Dockerfile`: MLflow 2.11 tracking server with PyMySQL and cloud storage drivers
  - `.env`: Generated environment variables for MLflow tracking URI and database connection
- **`ml/`**: Machine Learning Python Development Workspace
  - `Dockerfile`: Python 3.11 image with `uv`, PyTorch CPU, HuggingFace (`transformers`, `datasets`), and MLflow client
  - `.env`: Generated volume mounts and tracking URIs
- **`rustfs/`**: Lightweight Artifact Storage Service
  - `Dockerfile`: Multi-stage Actix-Web binary serving static files and artifacts
  - `.env`: Generated port and storage configuration
- **`.env`**: Consolidated environment file for root Docker Compose
