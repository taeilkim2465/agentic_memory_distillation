#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
bash "${SCRIPT_DIR}/exp_7_wf_subtask.sh" \
    --student-model "qwen3-8b-FC-vllm" \
    --memory-date   "0516" \
    --vllm-url      "${QWEN3_8B_VLLM_BASE_URL:-http://localhost:8771/v1}" \
    "$@"
