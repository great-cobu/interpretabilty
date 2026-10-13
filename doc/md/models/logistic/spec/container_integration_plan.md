# Containerization & MLOps Integration Plan (`models.logistic`)

## 1. Objective

Integrate the **Multinomial Logistic Regression (`models.logistic`)** glass-box baseline (alongside `models.MNIST`) with the project's containerized MLOps stack (**MariaDB**, **RustFS S3**, **MLflow Tracking Server**, and the **ML Workload / Marimo Container**) so that:
1. **SWI-Prolog (`prolog/*.pl`)** acts as the single source of truth for both **container infrastructure** and **model/dataset selection** (`model_spec/5`, `dataset_spec/6`, `compatible/2`, `experiment_spec/7`, `active_experiment/1`), generating `docker-compose.yml`, `docker/*/.env`, and `configs/generated/*.yaml`.
2. `models.logistic` training and evaluation run inside the `ml` container driven by the active Prolog-selected experiment (`$ACTIVE_CLI`, `$ACTIVE_CONFIG`), logging hyperparameters, epoch metrics, model checkpoints, and glass-box interpretability visualizations (`class_templates.png`, `learning_curves.png`, `confusion_matrix.png`, `sample_attributions.png`) to **MLflow** (`http://mlflow:5000`) backed by **MariaDB** and **RustFS**.
3. Automated tests verify the Prolog relational invariants (`./scripts/test.sh`), unit/integration tests inside the container (`pytest`), and end-to-end MLflow artifact persistence.
4. Results are immediately viewable in both the **MLflow Tracking UI** (`http://localhost:5000`) and the **Marimo Interactive Demo** (`http://localhost:2718`).

---

## 2. Current Gaps Identified

| Component | Current State | Required Change |
| :--- | :--- | :--- |
| **Prolog Policy Engine** ([`prolog/config.pl`](file:///Users/frankelliott/Documents/Projects/Python/interpetability/prolog/config.pl), [`prolog/gen_compose.pl`](file:///Users/frankelliott/Documents/Projects/Python/interpetability/prolog/gen_compose.pl), [`prolog/gen_env.pl`](file:///Users/frankelliott/Documents/Projects/Python/interpetability/prolog/gen_env.pl)) | Hardcodes Python `3.11`; mounts `hf_cache`; does not model available ML models, datasets, or experiment configs. | Update to Python `3.14`, remove `hf_cache`, expose Marimo port `2718`, add relational model/dataset catalog (`model_spec`, `dataset_spec`, `compatible`, `experiment_spec`, `active_experiment`), and add `prolog/gen_configs.pl`. |
| **ML Container Image** ([`docker/ml/Dockerfile`](file:///Users/frankelliott/Documents/Projects/Python/interpetability/docker/ml/Dockerfile)) | Uses `pip install -e .` without `uv`, sets obsolete `HF_HOME` / `TRANSFORMERS_CACHE` env vars, and runs `import transformers` in `HEALTHCHECK`. | Install `uv`, sync project dependencies (`--extra mlflow --group notebooks --group dev`), remove HuggingFace references, and update `HEALTHCHECK` to verify `torch`, `lightning`, `mlflow`, `models.logistic`, and `swipl`. |
| **MLflow Server Image** ([`docker/mlflow/Dockerfile`](file:///Users/frankelliott/Documents/Projects/Python/interpetability/docker/mlflow/Dockerfile)) | Pins `mlflow==2.11.1`, whereas `pyproject.toml` specifies `mlflow>=3.10.1`. | Align `docker/mlflow/Dockerfile` MLflow version with `pyproject.toml` (`mlflow>=3.10.1`) so client and server APIs match. |
| **Model Tracking Hooks** ([`src/models/logistic/train.py`](file:///Users/frankelliott/Documents/Projects/Python/interpetability/src/models/logistic/train.py), [`src/models/logistic/evaluate.py`](file:///Users/frankelliott/Documents/Projects/Python/interpetability/src/models/logistic/evaluate.py)) | `LogisticRunConfig.tracking` defines `enabled`, `uri`, and `experiment_name`, but `train.py` only instantiates `CSVLogger`. | Attach Lightning's `MLFlowLogger` when `config.tracking.enabled=True`, and log `config.yaml`, `learning_curves.png`, `class_templates.png`, `confusion_matrix.png`, `sample_attributions.png`, `training_summary.json`, `metrics.json`, and checkpoints to MLflow. |
| **Container & E2E Tests** ([`scripts/test.sh`](file:///Users/frankelliott/Documents/Projects/Python/interpetability/scripts/test.sh), `tests/`) | `scripts/test.sh` only checks legacy Prolog syntax and `.env` / `docker-compose.yml` strings. | Extend `scripts/test.sh` for model/dataset policy checks and add `scripts/test_containers.sh` to run an end-to-end containerized train + evaluate + MLflow verification cycle using `$ACTIVE_CLI` and `$ACTIVE_CONFIG`. |

---

## 3. Architecture & Data Flow

```mermaid
flowchart LR
    subgraph Host["Host Machine (macOS)"]
        Browser["Browser\n• MLflow UI (:5000)\n• RustFS Console (:9001)\n• Marimo Demo (:2718)"]
        Prolog["SWI-Prolog Policy & Registry\nprolog/config.pl\ngen_env.pl | gen_compose.pl | gen_configs.pl"]
        Workspace["Shared Workspace\n./configs/generated, ./runs, ./data"]
    end

    subgraph Docker["Docker Compose Network (mlops-net)"]
        ML["ml Container (Python 3.14 + uv + SWI-Prolog)\n• $ACTIVE_CLI train/test ($ACTIVE_CONFIG)\n• pytest\n• marimo run $ACTIVE_NOTEBOOK (:2718)"]
        MLflow["mlflow Container (:5000)\nMLflow Tracking Server"]
        MariaDB["mariadb Container (:3306)\nParams, Metrics & Run Metadata"]
        RustFS["rustfs Container (:9000 / :9001)\nS3 Artifact Bucket (s3://mlflow)"]
    end

    Prolog -->|Generates compose, .env, YAML configs| Docker
    Prolog -->|Generates configs/generated/*.yaml| Workspace
    Workspace <-->|Bind Mount /workspace| ML
    ML -->|MLFlowLogger & log_artifact| MLflow
    ML -->|S3 Artifact Upload (boto3)| RustFS
    MLflow -->|SQLAlchemy / PyMySQL| MariaDB
    MLflow -->|S3 Artifact Store| RustFS
    Browser --> MLflow
    Browser --> RustFS
    Browser --> ML
```

---

## 4. Phased Implementation Plan

### Phase 1: Prolog Model/Dataset Registry & Compose/Config Generation
**Goal**: Update `prolog/` so it declaratively governs models, datasets, compatibility, YAML run configs, `.env` files, and `docker-compose.yml` (see full specification in [`doc/md/prolog/spec/prolog_container_spec.md`](file:///Users/frankelliott/Documents/Projects/Python/interpetability/doc/md/prolog/spec/prolog_container_spec.md)).

1. **[`prolog/config.pl`](file:///Users/frankelliott/Documents/Projects/Python/interpetability/prolog/config.pl)**:
   - Update `global_config(python_version, '3.14')`, remove `required_volume(hf_cache)`, and add Marimo port `2718` to `ml`.
   - Define `model_spec/5` (`logistic` and `mnist_cnn`), `dataset_spec/6` (`mnist`), `compatible/2`, `experiment_spec/7` (`logistic_container`, `mnist_cnn_container`), and `active_experiment(logistic_container)`.
   - Add `validate_experiments/0` to `validate_config/0` so incompatible `(Model, Dataset)` pairs or invalid hyperparameters fail fast at generation time.
2. **[`prolog/gen_configs.pl`](file:///Users/frankelliott/Documents/Projects/Python/interpetability/prolog/gen_configs.pl)** (New Generator):
   - Generate `configs/generated/logistic_container.yaml` (and `configs/generated/mnist_cnn_container.yaml`) directly from `experiment_spec/7`, `dataset_spec/6`, `model_spec/5`, and the `mlflow` service URI (`http://mlflow:5000`).
3. **[`prolog/gen_env.pl`](file:///Users/frankelliott/Documents/Projects/Python/interpetability/prolog/gen_env.pl)** & **[`prolog/gen_compose.pl`](file:///Users/frankelliott/Documents/Projects/Python/interpetability/prolog/gen_compose.pl)**:
   - Export `ACTIVE_EXPERIMENT`, `ACTIVE_MODEL`, `ACTIVE_DATASET`, `ACTIVE_CLI`, `ACTIVE_CONFIG`, `ACTIVE_RUN_DIR`, `ACTIVE_NOTEBOOK`, `MARIMO_PORT`, and S3 endpoint variables (`MLFLOW_S3_ENDPOINT_URL=http://rustfs:9000`, `AWS_ENDPOINT_URL=http://rustfs:9000`) to `docker/ml/.env` and `docker/.env`.
   - Expose port `2718:2718` on `ml` and remove `hf_cache` from `docker-compose.yml`.
4. **[`scripts/gen_all.sh`](file:///Users/frankelliott/Documents/Projects/Python/interpetability/scripts/gen_all.sh)** & **[`scripts/test.sh`](file:///Users/frankelliott/Documents/Projects/Python/interpetability/scripts/test.sh)**:
   - Invoke `gen_configs.pl` alongside `gen_env.pl` and `gen_compose.pl`, and validate all Prolog rules and generated files.

---

### Phase 2: Container Image Modernization (`docker/ml` & `docker/mlflow`)
**Goal**: Ensure fast, reproducible builds and healthy containers for the MLOps stack.

1. **[`docker/ml/Dockerfile`](file:///Users/frankelliott/Documents/Projects/Python/interpetability/docker/ml/Dockerfile)**:
   - Base image: `python:${PYTHON_VERSION}-slim` (`3.14-slim`).
   - Install `uv` (`COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/`).
   - Install system dependencies (`curl`, `git`, `build-essential`, `swi-prolog-nox`).
   - Install project dependencies using `uv` (`--extra mlflow --group dev --group notebooks`).
   - Update `HEALTHCHECK` to validate:
     ```bash
     python -c "import torch, lightning, torchmetrics, mlflow, models; from models.logistic import LogisticClassifier" && swipl --version
     ```
2. **[`docker/mlflow/Dockerfile`](file:///Users/frankelliott/Documents/Projects/Python/interpetability/docker/mlflow/Dockerfile)**:
   - Upgrade `mlflow`, `pymysql`, `cryptography`, and `boto3` to match the client version in `pyproject.toml` (`mlflow>=3.10.1`).

---

### Phase 3: MLflow Tracking & Interpretability Artifact Logging in `models.logistic`
**Goal**: Make `mnist-logistic train` and `mnist-logistic test` log all metrics, parameters, checkpoints, and interpretability visualizations to MLflow when `tracking.enabled: true`.

1. **Interpretability Plot Generation (`src/models/logistic/evaluate.py` & `src/models/logistic/train.py`)**:
   - `save_class_templates_plot(model: LogisticClassifier, output_path: Path) -> Path`: renders the 10 learned $28 \times 28$ weight templates ($W_0 \dots W_9$) with a symmetric diverging colormap (`RdBu_r`) to `artifacts/class_templates.png`.
   - `save_confusion_matrix_plot(confusion_matrix: list[list[int]], output_path: Path) -> Path`: renders the $10 \times 10$ test confusion matrix to `evaluation/confusion_matrix.png`.
   - `save_sample_attributions_plot(model: LogisticClassifier, datamodule: MNISTDataModule, output_path: Path) -> Path`: renders sample test digits alongside their exact pixel attribution heatmaps ($x \odot W_c$) to `evaluation/sample_attributions.png`.
2. **MLflow Logger & Artifact Upload (`src/models/logistic/train.py` & `src/models/logistic/evaluate.py`)**:
   - When `config.tracking.enabled` is `True`:
     - Configure `MLFlowLogger(experiment_name=config.tracking.experiment_name, tracking_uri=config.tracking.uri, run_name=config.run_name, log_model=False)` alongside `CSVLogger`.
     - Persist `mlflow_run_id` in `artifacts/training_summary.json`.
     - Upload training artifacts (`config.yaml`, `artifacts/learning_curves.png`, `artifacts/class_templates.png`, `artifacts/training_summary.json`, `checkpoints/manifest.json`) to MLflow.
     - During `evaluate_run()`, log `test/loss` and `test/accuracy` metrics and upload evaluation artifacts (`evaluation/metrics.json`, `evaluation/confusion_matrix.png`, `evaluation/sample_attributions.png`) to the same MLflow run.

---

### Phase 4: End-to-End Container Verification & Automated Testing
**Goal**: Verify that containers build, pass healthchecks, execute the Prolog-selected active experiment, and persist artifacts to MariaDB + RustFS + MLflow.

1. **Unit & Integration Tests (`tests/test_logistic.py`)**:
   - Add tests for `save_class_templates_plot`, `save_confusion_matrix_plot`, `save_sample_attributions_plot`, and MLflow tracking artifact logging.
2. **Container Integration Script (`scripts/test_containers.sh`)**:
   - Step 1: Run `./scripts/gen_all.sh` and `./scripts/test.sh` (Prolog policy + model/dataset compatibility verification).
   - Step 2: Run `docker compose up -d --build` and wait for `mariadb`, `rustfs`, `mlflow`, and `ml` healthchecks to report `healthy`.
   - Step 3: Run pytest inside the `ml` container:
     ```bash
     docker compose exec -T ml pytest -q
     ```
   - Step 4: Execute containerized training and evaluation driven by the Prolog-generated environment variables (`$ACTIVE_CLI`, `$ACTIVE_CONFIG`, `$ACTIVE_RUN_DIR`):
     ```bash
     docker compose exec -T ml sh -c '$ACTIVE_CLI train --config $ACTIVE_CONFIG --overwrite && $ACTIVE_CLI test --run-dir $ACTIVE_RUN_DIR'
     ```
   - Step 5: Query the MLflow API (`http://mlflow:5000`) from inside `ml` to assert that the experiment run exists, `test/accuracy >= 0.90`, and all interpretability artifacts (`class_templates.png`, `learning_curves.png`, `confusion_matrix.png`, `sample_attributions.png`) are stored in the artifact repository.

---

### Phase 5: Displaying Results to the Principal Investigator
Once Phase 4 completes, results are simultaneously available across three interfaces:

1. **MLflow Tracking UI (`http://localhost:5000`)**:
   - Experiment: `mnist-logistic-interpretability`
   - Run: `logistic-container-demo`
   - **Metrics tab**: Interactive epoch curves for `train/loss`, `train/accuracy`, `val/loss`, `val/accuracy`, plus final `test/accuracy` (~92.5%) and `test/loss`.
   - **Artifacts tab**: Direct in-browser preview of:
     - `class_templates.png` (learned $28 \times 28$ weight templates $W_c$)
     - `learning_curves.png` (training & validation loss/accuracy curves)
     - `confusion_matrix.png` ($10 \times 10$ test confusion matrix)
     - `sample_attributions.png` (exact per-pixel attribution maps $x \odot W_c$)
     - `config.yaml`, `training_summary.json`, and `metrics.json`
2. **Marimo Interactive Interpretability Demo (`http://localhost:2718`)**:
   - Because `./runs` is bind-mounted into `/workspace/runs`, the container-trained run (`runs/logistic-container-demo`) appears automatically in the Marimo run selector dropdown.
3. **RustFS Object Storage Console (`http://localhost:9001`)**:
   - Inspect the underlying `mlflow` S3 bucket storing the logged artifacts.

---

## 5. Verification Checklist

- [ ] `./scripts/gen_all.sh` regenerates `docker/*/.env`, `docker-compose.yml`, and `configs/generated/*.yaml` from `prolog/*.pl`.
- [ ] `./scripts/test.sh` passes all Prolog infrastructure, model/dataset compatibility, and generated-file invariant tests.
- [ ] `docker compose ps` shows `mariadb`, `rustfs`, `mlflow`, and `ml` all in `Up (healthy)` status.
- [ ] `docker compose exec -T ml pytest` passes all unit and integration tests.
- [ ] `docker compose exec -T ml sh -c '$ACTIVE_CLI train --config $ACTIVE_CONFIG --overwrite && $ACTIVE_CLI test --run-dir $ACTIVE_RUN_DIR'` succeeds and populates both `runs/logistic-container-demo/` and the MLflow server at `http://localhost:5000`.
- [ ] Both `http://localhost:5000` (MLflow UI) and `http://localhost:2718` (Marimo Demo) display the trained logistic regression results and interpretability visualizations.
