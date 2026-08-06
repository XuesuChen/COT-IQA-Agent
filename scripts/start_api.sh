#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/.."
  pwd
)"

cd "$PROJECT_ROOT"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONUNBUFFERED=1

HOST="${API_HOST:-0.0.0.0}"
PORT="${API_PORT:-8000}"

echo "========================================"
echo " COT-IQA-Agent FastAPI"
echo "========================================"
echo "Project: $PROJECT_ROOT"
echo "API:     http://$HOST:$PORT"
echo "Swagger: http://$HOST:$PORT/docs"
echo "Environment: ${CONDA_DEFAULT_ENV:-unknown}"
echo "HF offline: $HF_HUB_OFFLINE"
echo
echo "Press Ctrl+C to stop."
echo

exec uvicorn app:app \
  --host "$HOST" \
  --port "$PORT" \
  --log-level info
