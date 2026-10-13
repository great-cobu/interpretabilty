#!/usr/bin/env bash
# Generate .env files, docker-compose.yml, and configs/generated/*.yaml from prolog/config.pl
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "==> Generating environment (.env) files..."
swipl -q -t main -f prolog/gen_env.pl

echo "==> Generating docker-compose.yml..."
swipl -q -t main -f prolog/gen_compose.pl

echo "==> Generating experiment YAML configs (configs/generated/)..."
swipl -q -t main -f prolog/gen_configs.pl

echo "==> Done! Infrastructure and experiment configurations up to date."
