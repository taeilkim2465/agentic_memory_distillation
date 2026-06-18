#!/usr/bin/env bash
# Exp 6 (wf+fn+st, system_replace) — all 4 models with per-model memory dirs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP6="${SCRIPT_DIR}/exp_6_st_dynamic_v2.sh"

TEST_CATEGORY="multi_turn_base"
NUM_THREADS=16
RESULT_TAG="0525"

# vLLM endpoint per model (env vars override defaults)
QWEN3_4B_URL="${QWEN3_4B_VLLM_BASE_URL:-http://localhost:8001/v1}"
QWEN3_8B_URL="${QWEN3_8B_VLLM_BASE_URL:-http://localhost:8000/v1}"
GEMMA4_E4B_URL="${GEMMA4_E4B_VLLM_BASE_URL:-http://localhost:8001/v1}"
LLAMA31_8B_URL="${LLAMA31_8B_VLLM_BASE_URL:-http://localhost:8000/v1}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-threads)   NUM_THREADS="$2";   shift 2 ;;
        --result-tag)    RESULT_TAG="$2";    shift 2 ;;
        --test-category) TEST_CATEGORY="$2"; shift 2 ;;
        *) echo "[ERROR] Unknown option: $1"; exit 1 ;;
    esac
done

declare -A MODEL_MEMORY=(
    ["qwen3-4b-FC-vllm"]="0523_qwen3_4b"
    ["qwen3-8b-FC-vllm"]="0523_qwen3_8b"
    ["gemma4-e4b-FC-vllm"]="0523_gemma4_e4b"
    ["llama3.1-8b-FC-vllm"]="0525_llama3_1_8b"
)

declare -A MODEL_URL=(
    ["qwen3-4b-FC-vllm"]="${QWEN3_4B_URL}"
    ["qwen3-8b-FC-vllm"]="${QWEN3_8B_URL}"
    ["gemma4-e4b-FC-vllm"]="${GEMMA4_E4B_URL}"
    ["llama3.1-8b-FC-vllm"]="${LLAMA31_8B_URL}"
)

MODELS=(
    "qwen3-4b-FC-vllm"
    "qwen3-8b-FC-vllm"
    "gemma4-e4b-FC-vllm"
    "llama3.1-8b-FC-vllm"
)

for MODEL in "${MODELS[@]}"; do
    MEM="${MODEL_MEMORY[$MODEL]}"
    URL="${MODEL_URL[$MODEL]}"
    echo ""
    echo "###############################################"
    echo "  Model      : ${MODEL}"
    echo "  Memory     : ${MEM}"
    echo "  vLLM URL   : ${URL}"
    echo "###############################################"
    bash "${EXP6}" \
        --student-model  "${MODEL}" \
        --memory-date    "${MEM}" \
        --vllm-url       "${URL}" \
        --test-category  "${TEST_CATEGORY}" \
        --num-threads    "${NUM_THREADS}" \
        --result-tag     "${RESULT_TAG}"
done

echo ""
echo "All 4 models done."
