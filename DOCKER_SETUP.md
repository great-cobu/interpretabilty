# Docker Setup for MLOps by Policy (Prolog + MLFlow)

This repository provides a containerized ML experiment tracking and execution stack managed declaratively through **SWI-Prolog** configuration rules.

---

## 1. Architecture

The stack consists of four containerized services orchestrated by `docker-compose.yml`, configured declaratively via Prolog policy rules:

```
+-------------------------------------------------------------+
|          Prolog Policy Engine (prolog/config.pl)            |
|           Generates .env and docker-compose.yml             |
+------------------------------+------------------------------+
                               |
                               v
                 Docker Compose Orchestration
                               |
  +-----------------+----------+----------+-----------------+
  |                 |                     |                 |
  v                 v                     v                 v
MariaDB           MLflow               RustFS          ML Workspace
(Database)    (Tracking Server)     (Artifacts)    (Python/PyTorch/HF)
```

* **MLflow** (`port 5050`): Experiment tracking server, UI, and REST API (uses `5050` to avoid macOS AirPlay Receiver on `5000`).
* **MariaDB** (`port 3306`): Relational backend metadata store for experiments, runs, parameters, and metrics.
* **RustFS** (`port 8088`): High-performance Actix-Web file and artifact storage service (uses `8088` to avoid Calibre on `8080`).
* **ML** (`port 2718`): Isolated Python 3.14 environment equipped with `uv`, SWI-Prolog (`swipl`), PyTorch, Lightning, Marimo, and MLflow client with `/workspace` bind-mounted.

---

## 2. Prerequisites

* Docker (v20.10+) and Docker Compose (v2.0+)
* SWI-Prolog (v8.4+ / v9.x / v10.x)

---

## 3. Configuration Management with Prolog

All configurations are defined declaratively in [`prolog/config.pl`](prolog/config.pl). Prolog generates both the service environment files (`docker/*/.env`), [`docker-compose.yml`](docker-compose.yml), and experiment configs (`configs/generated/*.yaml`).

### Quick Generation

```bash
# Regenerate .env files, docker-compose.yml, and experiment YAML configs
./scripts/gen_all.sh

# Or generate individually:
./scripts/gen_env.sh        # Generates docker/*/.env files
./scripts/gen_compose.sh    # Generates docker-compose.yml
```

### Direct SWI-Prolog Generation

```bash
# Generate .env files
swipl -q -t main -f prolog/gen_env.pl

# Generate docker-compose.yml
swipl -q -t main -f prolog/gen_compose.pl

# Generate experiment YAML configs
swipl -q -t main -f prolog/gen_configs.pl
```

### Configuration Structure in `prolog/config.pl`

* **Global Settings**:
  * `global_config(project_name, 'mlops_by_policy')` - Docker Compose project name
  * `global_config(python_version, '3.14')` - Python version for MLflow and ML services
  * `global_config(network_name, 'mlops_network')` - Docker bridge network
  * `global_config(base_path, Path)` - Root path for persistent data (configurable via `MLOPS_STORAGE_BASE` or `PROLOG_STORAGE_BASE` env vars, with automatic fallbacks for macOS/Linux/local directory)
* **MariaDB**:
  * `service_config(mariadb, user, 'mlflow_user')`
  * `service_config(mariadb, password, 'mlflow_pass_change_me')`
  * `service_config(mariadb, database, 'mlflow_db')`
  * `service_config(mariadb, port, 3306)`
* **MLflow**:
  * `service_config(mlflow, port, 5050)`
* **RustFS**:
  * `service_config(rustfs, port, 8088)`
  * `service_config(rustfs, data_dir, '/data')`
* **ML**:
  * `service_config(ml, marimo_port, 2718)`
* **Storage Resource Declarations**:
  * `service_storage(mariadb, data)` -> resolves to `./mlops_volumes/mariadb`
  * `service_storage(rustfs, data)` -> resolves to `./mlops_volumes/rustfs`
  * `service_storage(mlflow, data)` -> resolves to `./mlops_volumes/mlflow`
  * `service_storage(ml, data)` -> resolves to `./mlops_volumes/ml`
  * `service_storage(ml, datasets)` -> resolves to `./mlops_volumes/ml/datasets`
  * `service_storage(ml, models)` -> resolves to `./mlops_volumes/ml/models`

---

## 4. Starting and Managing the Stack

### Start Services

```bash
docker compose up -d
```

Check running containers:
```bash
docker compose ps
```

### Service Health Checks & UI

* **Marimo Demo UI**: Visit [http://localhost:2718](http://localhost:2718)
* **MLflow UI**: Visit [http://localhost:5050](http://localhost:5050)
* **MLflow Health**: `curl http://localhost:5050/health`
* **RustFS Status**: `curl http://localhost:8088/status`
* **MariaDB**: `docker compose exec mariadb mariadb-admin -u root -p ping`

### Stop Services

```bash
# Stop containers (preserves volume data)
docker compose down

# Stop and remove all volumes (clean slate)
docker compose down -v
```

---

## 5. Using the ML Container Workspace

The `ml` service provides a containerized Python environment with PyTorch, Lightning, Marimo, and MLflow client pre-configured.

### Interactive Shell

```bash
docker compose exec ml bash
```

Inside the container:
```python
import mlflow
import torch

print("PyTorch Version:", torch.__version__)
print("MLflow Tracking URI:", mlflow.get_tracking_uri())

with mlflow.start_run(run_name="python_sample_run"):
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_metric("val_loss", 0.12)
```

---

## 6. Troubleshooting

### MLflow fails to connect to database on startup
MariaDB takes 5-10 seconds to initialize during its first boot. The Docker Compose file includes automatic health checks and restart policies to reconnect automatically.

### Port Conflicts
* Port `5000` (macOS AirPlay Receiver) -> avoided by using `5050` for MLflow.
* Port `8080` (Calibre) -> avoided by using `8088` for RustFS.
If another port is occupied:
1. Change the port in [`prolog/config.pl`](prolog/config.pl) (e.g. `service_config(mlflow, port, 5050).` or `service_config(rustfs, port, 8088).`).
2. Run `./scripts/gen_all.sh`.
3. Run `docker compose up -d`.

### Regenerating Configuration
Always edit [`prolog/config.pl`](prolog/config.pl) and run `./scripts/gen_all.sh`. Never edit `docker-compose.yml` or `.env` files directly.
