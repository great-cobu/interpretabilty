#!/usr/bin/env bash
# Generate environment (.env) files from config.pl
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

swipl -q -t main -f prolog/gen_env.pl
