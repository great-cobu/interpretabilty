#!/usr/bin/env bash
# Automated test script for Prolog Docker configuration
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "==> 1. Testing Prolog configuration & assertions..."
swipl -q -g "
    consult('prolog/config.pl'),
    assertion(global_config(project_name, 'mlops_by_policy')),
    assertion(global_config(network_name, 'mlops_network')),
    assertion(global_config(base_path, '/Volumes/Macintosh HD/mlops_volumes')),
    assertion(service_storage(mariadb, data)),
    assertion(service_storage(ml, datasets)),
    assertion(storage_config(rustfs, data, 'rustfs')),
    assertion(storage_config(mlflow, data, 'mlflow')),
    assertion(storage_config(ml, data, 'ml')),
    assertion(storage_config(ml, datasets, 'ml/datasets')),
    assertion(storage_config(ml, models, 'ml/models')),
    assertion(storage_path(mariadb, data, '/Volumes/Macintosh HD/mlops_volumes/mariadb')),
    assertion(storage_path(rustfs, data, '/Volumes/Macintosh HD/mlops_volumes/rustfs')),
    assertion(storage_path(mlflow, data, '/Volumes/Macintosh HD/mlops_volumes/mlflow')),
    assertion(storage_path(ml, data, '/Volumes/Macintosh HD/mlops_volumes/ml')),
    assertion(storage_path(ml, datasets, '/Volumes/Macintosh HD/mlops_volumes/ml/datasets')),
    assertion(storage_path(ml, models, '/Volumes/Macintosh HD/mlops_volumes/ml/models')),
    writeln('    ✓ All standard Prolog unit assertions passed!'),
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
echo "    ✓ .env files generated successfully!"

echo "==> 4. Testing docker-compose.yml generation..."
swipl -q -t main -f prolog/gen_compose.pl
echo "    ✓ docker-compose.yml generated successfully!"

if command -v docker >/dev/null 2>&1; then
    echo "==> 5. Validating Docker Compose configuration..."
    docker compose config > /dev/null
    echo "    ✓ docker-compose.yml is valid according to Docker Compose schema!"
fi

echo "==> All tests passed successfully!"
