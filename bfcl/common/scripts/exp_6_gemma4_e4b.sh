#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
bash "${SCRIPT_DIR}/exp_6_st_dynamic_v2.sh" \
    --student-model "gemma4-e4b-FC-vllm" \
    --memory-date   "0516" \
    --vllm-url      "${GEMMA4_E4B_VLLM_BASE_URL:-http://localhost:8001/v1}" \
    "$@"
