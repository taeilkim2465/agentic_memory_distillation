#!/usr/bin/env bash
# Experiment 2: Workflow memory only
set -euo pipefail

STUDENT_MODEL="gemma4-e4b-FC-vllm"
TEST_CATEGORY="multi_turn_base"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8001/v1}"
MEMORY_DATE="0516"
NUM_THREADS=10
TEMPERATURE=0.0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --student-model) STUDENT_MODEL="$2"; shift 2 ;;
    --vllm-url)      VLLM_BASE_URL="$2"; shift 2 ;;
    --memory-date)   MEMORY_DATE="$2";   shift 2 ;;
    --num-threads)   NUM_THREADS="$2";   shift 2 ;;
    --temperature)   TEMPERATURE="$2";   shift 2 ;;
    -*)  echo "[ERROR] Unknown option: $1"; exit 1 ;;
    *)   shift ;;
  esac
done

PYTHON="${BFCL_PYTHON:-python3}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MEMORY_DIR="${REPO_ROOT}/data/memory/${MEMORY_DATE}"

echo "========================================"
echo " Exp 2 (Gemma4): Workflow memory only"
echo "  Model      : ${STUDENT_MODEL}"
echo "  vLLM URL   : ${VLLM_BASE_URL}"
echo "  Memory dir : ${MEMORY_DIR}"
echo "  Threads    : ${NUM_THREADS}"
echo "========================================"

if [ ! -f "${MEMORY_DIR}/workflow/documents.json" ]; then
  echo "[ERROR] Workflow memory not found: ${MEMORY_DIR}/workflow/documents.json"
  exit 1
fi

export OPENAI_API_KEY="${OPENAI_API_KEY}"
export VLLM_BASE_URL VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"

cd "${REPO_ROOT}"
"${PYTHON}" -m bfcl_eval generate \
  --model "${STUDENT_MODEL}" \
  --test-category "${TEST_CATEGORY}" \
  --temperature "${TEMPERATURE}" \
  --num-threads "${NUM_THREADS}" \
  --memory-dir "${MEMORY_DIR}" \
  --memory-types "workflow" \
  --result-dir "result/gemma4_exp2_workflow/${STUDENT_MODEL}"
