#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
bash "${SCRIPT_DIR}/exp_6_st_dynamic_v2.sh" \
    --student-model "llama3.1-8b-FC-vllm" \
    --memory-date   "0516" \
    --vllm-url      "${LLAMA31_8B_VLLM_BASE_URL:-http://localhost:8000/v1}" \
    "$@"
