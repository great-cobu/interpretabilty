#!/usr/bin/env bash
# Automated test script for Prolog Docker & Experiment Policy configuration
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "==> 1. Testing Prolog configuration, model/dataset registry & assertions..."
swipl -q -g "
    consult('prolog/config.pl'),
    assertion(global_config(project_name, 'mlops_by_policy')),
    assertion(global_config(network_name, 'mlops_network')),
    assertion(global_config(python_version, '3.14')),
    assertion(global_config(base_path, './mlops_volumes')),
    assertion(service_storage(mariadb, data)),
    assertion(service_storage(ml, datasets)),
    assertion(storage_config(rustfs, data, 'rustfs')),
    assertion(storage_config(mlflow, data, 'mlflow')),
    assertion(storage_config(ml, data, 'ml')),
    assertion(storage_config(ml, datasets, 'ml/datasets')),
    assertion(storage_config(ml, models, 'ml/models')),
    assertion(storage_path(mariadb, data, './mlops_volumes/mariadb')),
    assertion(storage_path(rustfs, data, './mlops_volumes/rustfs')),
    assertion(storage_path(mlflow, data, './mlops_volumes/mlflow')),
    assertion(storage_path(ml, data, './mlops_volumes/ml')),
    assertion(storage_path(ml, datasets, './mlops_volumes/ml/datasets')),
    assertion(storage_path(ml, models, './mlops_volumes/ml/models')),
    assertion(reserved_host_port(8080, 'Calibre Server')),
    assertion(reserved_host_port(5000, 'macOS ControlCenter / AirPlay Receiver')),
    assertion(service_config(rustfs, port, 8088)),
    assertion(service_config(mlflow, port, 5050)),
    assertion(validate_ports),
    assertion(validate_experiments),
    assertion(compatible(logistic, mnist)),
    assertion(compatible(mnist_cnn, mnist)),
    assertion((
        active_experiment(ActiveExp),
        active_experiment_details(
            ActiveExp,
            _ModelId,
            mnist,
            _CliCommand,
            _ConfigPath,
            _RunDir,
            _NotebookPath,
            _ExperimentName
        )
    )),
    writeln('    ✓ All standard Prolog unit and model/dataset registry assertions passed!'),
    halt.
"

echo "==> 2. Testing MLOPS_STORAGE_BASE environment override..."
MLOPS_STORAGE_BASE="/tmp/test_mlops_volumes" swipl -q -g "
    consult('prolog/config.pl'),
    assertion(global_config(base_path, '/tmp/test_mlops_volumes')),
    assertion(storage_path(mariadb, data, '/tmp/test_mlops_volumes/mariadb')),
    assertion(storage_path(ml, datasets, '/tmp/test_mlops_volumes/ml/datasets')),
    writeln('    ✓ Environment override portability test passed!'),
    halt.
"

echo "==> 3. Testing .env file generation..."
swipl -q -t main -f prolog/gen_env.pl
grep -q "^ACTIVE_EXPERIMENT=" docker/ml/.env
grep -q "^ACTIVE_MODEL=" docker/ml/.env
grep -q "^ACTIVE_CLI=" docker/ml/.env
grep -q "^ACTIVE_CONFIG=configs/generated/" docker/ml/.env
echo "    ✓ .env files generated and verified successfully!"

echo "==> 4. Testing docker-compose.yml generation..."
swipl -q -t main -f prolog/gen_compose.pl
echo "    ✓ docker-compose.yml generated successfully!"

echo "==> 5. Testing experiment YAML generation..."
swipl -q -t main -f prolog/gen_configs.pl
test -f configs/generated/logistic_container.yaml
test -f configs/generated/mnist_cnn_container.yaml
echo "    ✓ configs/generated/*.yaml generated successfully!"

if command -v docker >/dev/null 2>&1; then
    echo "==> 6. Validating Docker Compose configuration..."
    docker compose config > /dev/null
    echo "    ✓ docker-compose.yml is valid according to Docker Compose schema!"
fi

echo "==> All Prolog policy & configuration tests passed successfully!"
