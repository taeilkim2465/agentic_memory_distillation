#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
bash "${SCRIPT_DIR}/exp_6_st_dynamic_v2.sh" \
    --student-model "llama3.1-8b-FC-vllm" \
    --memory-date   "0525_llama3_1_8b" \
    --vllm-url      "${LLAMA31_8B_VLLM_BASE_URL:-http://10.10.0.118:8000/v1}" \
    "$@"
