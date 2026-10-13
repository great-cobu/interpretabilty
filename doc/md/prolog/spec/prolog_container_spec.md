# Prolog Policy, Model/Dataset Registry & Container Generation Specification

## 1. Purpose

The `prolog/` directory serves as the **single source of truth** for both:
1. **Containerized MLOps Infrastructure** (`mariadb`, `rustfs`, `mlflow`, `ml` services, ports, volumes, and network endpoints).
2. **Declarative Model, Dataset & Experiment Registry** (which models exist, which datasets exist, which `(Model, Dataset)` pairs are compatible, and which experiment is active in the container pipeline).

Neither [`docker-compose.yml`](file:///Users/frankelliott/Documents/Projects/Python/interpetability/docker-compose.yml), `docker/*/.env`, nor `configs/generated/*.yaml` may be edited by hand; all of them are generated from [`prolog/config.pl`](file:///Users/frankelliott/Documents/Projects/Python/interpetability/prolog/config.pl) via:
- [`prolog/gen_env.pl`](file:///Users/frankelliott/Documents/Projects/Python/interpetability/prolog/gen_env.pl)
- [`prolog/gen_compose.pl`](file:///Users/frankelliott/Documents/Projects/Python/interpetability/prolog/gen_compose.pl)
- [`prolog/gen_configs.pl`](file:///Users/frankelliott/Documents/Projects/Python/interpetability/prolog/gen_configs.pl)

---

## 2. Relational Schema in [`prolog/config.pl`](file:///Users/frankelliott/Documents/Projects/Python/interpetability/prolog/config.pl)

### 2.1 Model Catalog (`model_spec/5`)
Declares each available model architecture, its Python module, its CLI entry point, its Marimo notebook path, and its MLflow experiment name:

```prolog
% model_spec(ModelId, Module, CliCommand, NotebookPath, ExperimentName)
model_spec(
    logistic,
    'models.logistic',
    'mnist-logistic',
    'notebooks/logistic_demo.py',
    'mnist-logistic-interpretability'
).
model_spec(
    mnist_cnn,
    'models.MNIST',
    'mnist-cnn',
    'notebooks/mnist_cnn_demo.py',
    'mnist-cnn-interpretability'
).
```

### 2.2 Dataset Catalog (`dataset_spec/6`)
Declares each dataset, its container-relative root directory, input channels, image dimensions, number of classes, and default validation split fraction:

```prolog
% dataset_spec(DatasetId, DataRoot, InChannels, ImageSize, NumClasses, ValSplit)
dataset_spec(mnist, './data', 1, 28, 10, 0.1).
```

### 2.3 Model–Dataset Compatibility Relation (`compatible/2`)
Explicitly constrains which models are valid for which datasets:

```prolog
% compatible(ModelId, DatasetId)
compatible(logistic, mnist).
compatible(mnist_cnn, mnist).
```

### 2.4 Experiment Definitions & Active Selection (`experiment_spec/7` & `active_experiment/1`)
Declares named experiment configurations and selects the active experiment for containerized execution and testing:

```prolog
% experiment_spec(ExpId, ModelId, DatasetId, RunName, Epochs, BatchSize, LearningRate)
experiment_spec(
    logistic_container,
    logistic,
    mnist,
    'logistic-container-demo',
    10,
    128,
    0.05
).
experiment_spec(
    mnist_cnn_container,
    mnist_cnn,
    mnist,
    'mnist-cnn-container-demo',
    5,
    64,
    0.001
).

% Active experiment driving the container environment and test pipeline
active_experiment(logistic_container).
```

### 2.5 Policy Validation Invariants (`validate_config/0`)
Extend `validate_config/0` in [`prolog/config.pl`](file:///Users/frankelliott/Documents/Projects/Python/interpetability/prolog/config.pl) with `validate_experiments/0` to enforce:
1. Every `active_experiment(ExpId)` references a defined `experiment_spec(ExpId, ModelId, DatasetId, _, Epochs, BatchSize, LR)`.
2. `ModelId` is a registered `model_spec/5` and `DatasetId` is a registered `dataset_spec/6`.
3. `compatible(ModelId, DatasetId)` holds (rejects incompatible model/dataset pairings at generation time).
4. Hyperparameters satisfy domain constraints (`Epochs > 0`, `BatchSize > 0`, `LR > 0`).
5. Data and notebook paths pass `valid_portable_path/1`.

---

## 3. Infrastructure Policy Updates ([`prolog/config.pl`](file:///Users/frankelliott/Documents/Projects/Python/interpetability/prolog/config.pl))

1. **Python Version**:
   ```prolog
   global_config(python_version, '3.14').
   ```
2. **ML Container & Marimo Port**:
   ```prolog
   service_config(ml, container_name, 'mlops-ml').
   service_config(ml, external_port, 2718).
   service_config(ml, internal_port, 2718).
   ```
3. **Volume Cleanup**:
   Remove `required_volume(hf_cache)` while preserving `mariadb_data`, `rustfs_data`, and the `./:/workspace` bind mount on `ml`.

---

## 4. Generators

### 4.1 YAML Experiment Config Generator ([`prolog/gen_configs.pl`](file:///Users/frankelliott/Documents/Projects/Python/interpetability/prolog/gen_configs.pl))
A new SWI-Prolog generator that queries `experiment_spec/7`, `model_spec/5`, `dataset_spec/6`, and `service_config(mlflow, ...)` to emit deterministic YAML configuration files under `configs/generated/` (e.g., `configs/generated/logistic_container.yaml` and `configs/generated/mnist_cnn_container.yaml`).

For `logistic_container`, `prolog/gen_configs.pl` generates `configs/generated/logistic_container.yaml`:
```yaml
# GENERATED BY SWI-PROLOG (prolog/gen_configs.pl) - DO NOT EDIT BY HAND
seed: 42
run_name: logistic-container-demo
output_root: ./runs
data:
  root: ./data
  batch_size: 128
  num_workers: 0
  val_split: 0.1
  download: true
training:
  epochs: 10
  learning_rate: 0.05
  weight_decay: 0.0001
  accelerator: auto
  devices: 1
  deterministic: true
  log_every_n_steps: 25
tracking:
  enabled: true
  uri: http://mlflow:5000
  experiment_name: mnist-logistic-interpretability
```

### 4.2 Environment Variable Generator ([`prolog/gen_env.pl`](file:///Users/frankelliott/Documents/Projects/Python/interpetability/prolog/gen_env.pl))
Queries `active_experiment/1` along with `experiment_spec/7` and `model_spec/5` so `docker/ml/.env` and `docker/.env` automatically export the active model, dataset, CLI, config path, and notebook path:
- `PYTHON_VERSION=3.14`
- `MLFLOW_TRACKING_URI=http://mlflow:5000`
- `MLFLOW_S3_ENDPOINT_URL=http://rustfs:9000`
- `AWS_ENDPOINT_URL=http://rustfs:9000`
- `ACTIVE_EXPERIMENT=logistic_container`
- `ACTIVE_MODEL=logistic`
- `ACTIVE_DATASET=mnist`
- `ACTIVE_CLI=mnist-logistic`
- `ACTIVE_CONFIG=configs/generated/logistic_container.yaml`
- `ACTIVE_RUN_DIR=runs/logistic-container-demo`
- `ACTIVE_NOTEBOOK=notebooks/logistic_demo.py`
- `MARIMO_PORT=2718`

### 4.3 Compose Generator ([`prolog/gen_compose.pl`](file:///Users/frankelliott/Documents/Projects/Python/interpetability/prolog/gen_compose.pl))
- Publishes `"${ML_EXTERNAL_PORT}:${ML_INTERNAL_PORT}"` (`"2718:2718"`) on the `ml` container.
- Removes `hf_cache` volume mappings.
- Preserves `service_healthy` dependencies on `mariadb`, `rustfs`, and `mlflow`.

---

## 5. Policy & Generation Workflow ([`scripts/gen_all.sh`](file:///Users/frankelliott/Documents/Projects/Python/interpetability/scripts/gen_all.sh) & [`scripts/test.sh`](file:///Users/frankelliott/Documents/Projects/Python/interpetability/scripts/test.sh))

1. `./scripts/gen_all.sh` runs:
   - `swipl -q -t main -f prolog/gen_env.pl`
   - `swipl -q -t main -f prolog/gen_compose.pl`
   - `swipl -q -t main -f prolog/gen_configs.pl`
2. `./scripts/test.sh` verifies:
   - Prolog compilation of `config.pl`, `gen_env.pl`, `gen_compose.pl`, and `gen_configs.pl`.
   - `validate_config` (including `validate_experiments` model/dataset compatibility checks).
   - Generated `.env`, `docker-compose.yml`, and `configs/generated/*.yaml` files exist and match the active Prolog experiment policy.
