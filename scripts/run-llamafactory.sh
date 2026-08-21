#!/usr/bin/env bash
# LlamaFactory pipeline launcher for launchd KeepAlive.
# Serves API + static UI on :8899 (single process).
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR" || exit 1

ulimit -n 8192 2>/dev/null || true

if [ -f ./.env.local ]; then
  set -a
  . ./.env.local
  set +a
fi

mkdir -p logs llamafactory_pipeline/assistant_state
export PYTHONUNBUFFERED=1
export PYTHONPATH=".:eval${PYTHONPATH:+:$PYTHONPATH}"
PORT="${LLAMAFACTORY_PORT:-8899}"

if [ -x .venv/bin/python ]; then
  PYTHON_BIN=.venv/bin/python
elif [ -x .venv-api/bin/python ]; then
  PYTHON_BIN=.venv-api/bin/python
else
  PYTHON_BIN=python3
fi

exec "$PYTHON_BIN" -m uvicorn llamafactory_pipeline.app:app \
  --host 127.0.0.1 --port "$PORT"
