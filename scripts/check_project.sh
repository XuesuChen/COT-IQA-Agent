#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/.."
  pwd
)"

cd "$PROJECT_ROOT"

echo "===== PYTHON ====="
python --version

echo
echo "===== CUDA ====="
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("cuda_device:", torch.cuda.get_device_name(0))
PY

echo
echo "===== CONFIGURATION ====="
python - <<'PY'
from pathlib import Path

from configs.config_loader import load_config

config = load_config()
cot = config.get("cot_iqa", {})
rag = config.get("rag", {})
paths = config.get("paths", {})

base_model = Path(
    str(cot.get("base_model_path", ""))
).expanduser()

adapter = Path(
    str(cot.get("adapter_path", ""))
).expanduser()

vector_store = Path(
    str(
        paths.get(
            "vector_store_dir",
            "rag/vector_store",
        )
    )
).expanduser()

required_index_files = (
    "index.faiss",
    "chunks.jsonl",
    "manifest.json",
)

checks = {
    "base_model": (
        base_model.is_dir()
        and (base_model / "config.json").is_file()
    ),
    "adapter": (
        adapter.is_dir()
        and (adapter / "adapter_config.json").is_file()
    ),
    "rag_index": (
        vector_store.is_dir()
        and all(
            (vector_store / name).is_file()
            for name in required_index_files
        )
    ),
    "rag_runtime_cpu": (
        str(rag.get("runtime_device", "")).lower()
        == "cpu"
    ),
}

for name, passed in checks.items():
    print(
        f"{'[PASS]' if passed else '[FAIL]'} "
        f"{name}"
    )

failed = [
    name
    for name, passed in checks.items()
    if not passed
]

if failed:
    raise SystemExit(
        f"Project check failed: {failed}"
    )

print("Project resource check: PASSED")
PY

echo
echo "===== PYTHON SYNTAX ====="

python -m py_compile \
  app.py \
  ui/gradio_app.py \
  agent/state.py \
  agent/router.py \
  agent/graph.py \
  agent/nodes.py \
  rag/build_index.py \
  rag/retriever.py

echo "Python syntax check: PASSED"

echo
echo "===== API HEALTH ====="

python - <<'PY'
from app import health_check

health = health_check()

assert health.status == "ok"
assert health.cuda_available
assert health.cot_iqa_model_configured
assert health.rag_index_ready
assert health.report_directory_ready

print("FastAPI health check: PASSED")
PY

echo
echo "========================================"
echo " COT-IQA-Agent project check: PASSED"
echo "========================================"
