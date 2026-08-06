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

HOST="${GRADIO_SERVER_NAME:-0.0.0.0}"
PORT="${GRADIO_SERVER_PORT:-7860}"

echo "========================================"
echo " COT-IQA-Agent Gradio"
echo "========================================"
echo "Project: $PROJECT_ROOT"
echo "Address: http://$HOST:$PORT"
echo "Environment: ${CONDA_DEFAULT_ENV:-unknown}"
echo "HF offline: $HF_HUB_OFFLINE"
echo
echo "Press Ctrl+C to stop."
echo

exec python ui/gradio_app.py
