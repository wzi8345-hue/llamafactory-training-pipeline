#!/usr/bin/env bash
set -euo pipefail

export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

if [ -f ./.env.local ]; then
  set -a
  . ./.env.local
  set +a
fi

mkdir -p logs llamafactory_pipeline/assistant_state
export PYTHONUNBUFFERED=1
export PYTHONPATH=".:eval${PYTHONPATH:+:$PYTHONPATH}"

if [ -x .venv/bin/python ]; then
  PYTHON_BIN=.venv/bin/python
elif [ -x .venv-api/bin/python ]; then
  PYTHON_BIN=.venv-api/bin/python
else
  PYTHON_BIN=python3
fi

exec "$PYTHON_BIN" -m llamafactory_pipeline.assistant_worker --once --limit 20
