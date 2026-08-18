#!/usr/bin/env bash
# Generate both .env files and docker-compose.yml from config.pl
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "==> Generating environment (.env) files..."
swipl -q -t main -f prolog/gen_env.pl

echo "==> Generating docker-compose.yml..."
swipl -q -t main -f prolog/gen_compose.pl

echo "==> Done! Infrastructure configuration up to date."
