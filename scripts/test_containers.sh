#!/usr/bin/env bash
# End-to-end containerization & MLOps verification script
# Verifies Prolog policy -> Docker build & healthchecks -> pytest -> training/eval -> MLflow/RustFS/Marimo
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "==> Step 1: Regenerating and validating Prolog policy & experiment configs..."
./scripts/gen_all.sh
./scripts/test.sh

# Source generated docker/.env so we can pre-create host bind-mount volume directories
set -a
source docker/.env
set +a

echo "==> Step 2: Ensuring persistent storage directories exist under ${STORAGE_BASE_PATH}..."
mkdir -p "${STORAGE_MARIADB_PATH}" \
         "${STORAGE_RUSTFS_PATH}" \
         "${STORAGE_MLFLOW_PATH}/.cache" \
         "${STORAGE_ML_PATH}" \
         "${STORAGE_ML_DATASETS_PATH}" \
         "${STORAGE_ML_MODELS_PATH}"

echo "==> Step 3: Building and starting containers (mariadb, rustfs, mlflow, ml)..."
docker compose up -d --build

echo "==> Step 4: Waiting for all 4 services (mariadb, rustfs, mlflow, ml) to become healthy..."
for service in mariadb rustfs mlflow ml; do
    container="mlops_${service}"
    echo -n "    Waiting for ${container}..."
    for _ in $(seq 1 60); do
        status="$(docker inspect --format='{{.State.Health.Status}}' "${container}" 2>/dev/null || echo "starting")"
        if [ "${status}" = "healthy" ]; then
            echo " healthy!"
            break
        fi
        sleep 2
    done
    final_status="$(docker inspect --format='{{.State.Health.Status}}' "${container}" 2>/dev/null || echo "unknown")"
    if [ "${final_status}" != "healthy" ]; then
        echo " ERROR: ${container} did not become healthy (status=${final_status})"
        docker compose logs "${service}"
        exit 1
    fi
done

echo "==> Step 5: Running unit & integration test suite inside mlops_ml container..."
docker compose exec -T ml pytest -q

echo "==> Step 6: Running Prolog-selected experiment (${ACTIVE_EXPERIMENT}: ${ACTIVE_CLI} with ${ACTIVE_CONFIG})..."
rm -rf "${ACTIVE_RUN_DIR}"
docker compose exec -T ml sh -c '
    set -e
    echo "    Training model=${ACTIVE_MODEL} on dataset=${ACTIVE_DATASET}..."
    ${ACTIVE_CLI} train --config "${ACTIVE_CONFIG}"
    echo "    Evaluating best checkpoint on test split..."
    ${ACTIVE_CLI} test \
        --config "${ACTIVE_CONFIG}" \
        --checkpoint "${ACTIVE_RUN_DIR}/checkpoints/best.ckpt" \
        --output-dir "${ACTIVE_RUN_DIR}/evaluation"
'

echo "==> Step 7: Verifying MLflow Tracking Server, RustFS artifacts, and Marimo UI..."
docker compose exec -T ml python - << 'PYEOF'
import json
import os
from pathlib import Path
import urllib.request
import mlflow

tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
exp_name = os.environ["ACTIVE_MLFLOW_EXPERIMENT"]
run_dir = Path(os.environ["ACTIVE_RUN_DIR"])

summary = json.loads((run_dir / "training_summary.json").read_text())
metrics = json.loads((run_dir / "evaluation" / "metrics.json").read_text())
run_id = summary["mlflow_run_id"]
assert run_id, "Expected mlflow_run_id in training_summary.json"

client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
run = client.get_run(run_id)
test_acc = run.data.metrics.get("test/accuracy", 0.0)
val_acc = run.data.metrics.get("val/accuracy", 0.0)
assert test_acc >= 0.90, f"Expected test/accuracy >= 0.90, got {test_acc}"

train_artifacts = {a.path for a in client.list_artifacts(run_id, "training")}
eval_artifacts = {a.path for a in client.list_artifacts(run_id, "evaluation")}
assert "training/class_templates.png" in train_artifacts, f"Missing class_templates.png: {train_artifacts}"
assert "training/learning_curves.png" in train_artifacts, f"Missing learning_curves.png: {train_artifacts}"
assert "evaluation/confusion_matrix.png" in eval_artifacts, f"Missing confusion_matrix.png: {eval_artifacts}"
assert "evaluation/sample_attributions.png" in eval_artifacts, f"Missing sample_attributions.png: {eval_artifacts}"

# Verify RustFS health & Marimo server inside container
with urllib.request.urlopen("http://rustfs:8080/health", timeout=5) as resp:
    assert resp.status == 200
with urllib.request.urlopen("http://localhost:2718", timeout=5) as resp:
    assert resp.status == 200

print(f"    ✓ Verified MLflow Run ID:      {run_id}")
print(f"    ✓ Experiment Name:             {exp_name}")
print(f"    ✓ Validation Accuracy:         {val_acc:.4f}")
print(f"    ✓ Test Split Accuracy:         {test_acc:.4f} ({metrics['sample_count']} test digits)")
print(f"    ✓ Logged Training Artifacts:   {sorted(train_artifacts)}")
print(f"    ✓ Logged Evaluation Artifacts: {sorted(eval_artifacts)}")
print("    ✓ RustFS (:8080) and Marimo (:2718) HTTP endpoints verified!")
PYEOF

echo "==> All container services, tests, training, evaluation, and UI endpoints succeeded!"
echo "    • MLflow UI:      http://localhost:5000"
echo "    • Marimo Demo UI: http://localhost:2718"
echo "    • RustFS Storage: http://localhost:8080/artifacts/mlflow/"
