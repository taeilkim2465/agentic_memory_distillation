#!/usr/bin/env bash
# Experiment 1: Baseline (no memory)
set -euo pipefail

STUDENT_MODEL="qwen3-4b-FC-vllm"
TEST_CATEGORY="multi_turn_base"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8001/v1}"
NUM_THREADS=10
TEMPERATURE=0.0
RESULT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --student-model) STUDENT_MODEL="$2"; shift 2 ;;
    --vllm-url)      VLLM_BASE_URL="$2"; shift 2 ;;
    --test-category) TEST_CATEGORY="$2"; shift 2 ;;
    --num-threads)   NUM_THREADS="$2";   shift 2 ;;
    --temperature)   TEMPERATURE="$2";   shift 2 ;;
    --result-dir)    RESULT_DIR="$2";    shift 2 ;;
    -*)  echo "[ERROR] Unknown option: $1"; exit 1 ;;
    *)   shift ;;
  esac
done

PYTHON="${BFCL_PYTHON:-python3}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULT_DIR="${RESULT_DIR:-result/exp1_baseline/${STUDENT_MODEL}}"

echo "========================================"
echo " Exp 1: Baseline (no memory)"
echo "  Model    : ${STUDENT_MODEL}"
echo "  vLLM URL : ${VLLM_BASE_URL}"
echo "  Category : ${TEST_CATEGORY}"
echo "  Threads  : ${NUM_THREADS}"
echo "========================================"

export OPENAI_API_KEY="${OPENAI_API_KEY}"
export VLLM_BASE_URL VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"

cd "${REPO_ROOT}"
"${PYTHON}" -m bfcl_eval generate \
  --model "${STUDENT_MODEL}" \
  --test-category "${TEST_CATEGORY}" \
  --temperature "${TEMPERATURE}" \
  --num-threads "${NUM_THREADS}" \
  --result-dir "${RESULT_DIR}"
