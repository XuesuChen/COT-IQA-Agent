#!/usr/bin/env bash
set -uo pipefail

TARGET_PID="${1:-260554}"
PROJECT_DIR="/home/cxs/COT-IQA-Agent"
CONDA_BIN="/home/cxs/miniconda3/bin/conda"
MIN_FREE_MEMORY_MIB=20000

echo "[$(date '+%F %T')] Waiting for process PID ${TARGET_PID}."

while ps -p "${TARGET_PID}" > /dev/null 2>&1; do
    echo "[$(date '+%F %T')] PID ${TARGET_PID} is still running."
    sleep 60
done

echo "[$(date '+%F %T')] PID ${TARGET_PID} has finished."
echo "[$(date '+%F %T')] Waiting for at least ${MIN_FREE_MEMORY_MIB} MiB free GPU memory."

while true; do
    FREE_MEMORY="$(
        nvidia-smi \
            --query-gpu=memory.free \
            --format=csv,noheader,nounits \
        | head -n 1 \
        | tr -d ' '
    )"

    if [[ "${FREE_MEMORY}" =~ ^[0-9]+$ ]] \
        && (( FREE_MEMORY >= MIN_FREE_MEMORY_MIB )); then
        echo "[$(date '+%F %T')] GPU free memory: ${FREE_MEMORY} MiB."
        break
    fi

    echo "[$(date '+%F %T')] GPU free memory: ${FREE_MEMORY:-unknown} MiB; continuing to wait."
    sleep 60
done

cd "${PROJECT_DIR}" || exit 1

echo "[$(date '+%F %T')] Starting CoT-IQA model test."

"${CONDA_BIN}" run \
    -n cotagent \
    --no-capture-output \
    python -u scripts/smoke_test_cot_iqa.py

EXIT_CODE=$?

echo "[$(date '+%F %T')] Test process exited with code ${EXIT_CODE}."
exit "${EXIT_CODE}"
