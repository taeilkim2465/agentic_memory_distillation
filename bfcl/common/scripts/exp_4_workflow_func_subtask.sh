#!/usr/bin/env bash
# Experiment 4: Workflow + Error Function + Subtask memory (all)
set -euo pipefail

STUDENT_MODEL="qwen3-4b-FC-vllm"
DECOMPOSER_LLM=""
SUBTASK_SOURCE="turns"
TEST_CATEGORY="multi_turn_base"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8001/v1}"
MEMORY_DATE="0516"
NUM_THREADS=10
TEMPERATURE=0.0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --student-model)  STUDENT_MODEL="$2";   shift 2 ;;
    --vllm-url)       VLLM_BASE_URL="$2";   shift 2 ;;
    --test-category)  TEST_CATEGORY="$2";   shift 2 ;;
    --decomposer-llm) DECOMPOSER_LLM="$2";  shift 2 ;;
    --subtask-source) SUBTASK_SOURCE="$2";  shift 2 ;;
    --memory-date)    MEMORY_DATE="$2";     shift 2 ;;
    --num-threads)    NUM_THREADS="$2";     shift 2 ;;
    --temperature)    TEMPERATURE="$2";     shift 2 ;;
    -*)  echo "[ERROR] Unknown option: $1"; exit 1 ;;
    *)   shift ;;
  esac
done

# Derive decomposer from student model if not explicitly set (strip -FC-vllm suffix)
if [[ -z "$DECOMPOSER_LLM" ]]; then
  DECOMPOSER_LLM="openai/${STUDENT_MODEL%-FC-vllm}"
fi

PYTHON="${BFCL_PYTHON:-python3}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MEMORY_DIR="${REPO_ROOT}/data/memory/${MEMORY_DATE}"

echo "========================================"
echo " Exp 4: Workflow + Error Function + Subtask memory"
echo "  Model          : ${STUDENT_MODEL}"
echo "  vLLM URL       : ${VLLM_BASE_URL}"
echo "  Category       : ${TEST_CATEGORY}"
echo "  Subtask source : ${SUBTASK_SOURCE}"
echo "  Decomposer LLM : ${DECOMPOSER_LLM}"
echo "  Memory dir     : ${MEMORY_DIR}"
echo "  Threads        : ${NUM_THREADS}"
echo "========================================"

for F in "${MEMORY_DIR}/workflow/documents.json" \
         "${MEMORY_DIR}/subtask/segments.jsonl" \
         "${MEMORY_DIR}/function/records.jsonl"; do
  if [ ! -f "${F}" ]; then
    echo "[ERROR] Memory file not found: ${F}"
    exit 1
  fi
done

export OPENAI_API_KEY="${OPENAI_API_KEY}"
export VLLM_BASE_URL VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"

cd "${REPO_ROOT}"
"${PYTHON}" -m bfcl_eval generate \
  --model "${STUDENT_MODEL}" \
  --test-category "${TEST_CATEGORY}" \
  --temperature "${TEMPERATURE}" \
  --num-threads "${NUM_THREADS}" \
  --memory-dir "${MEMORY_DIR}" \
  --memory-types "workflow,function,subtask" \
  --decomposer-llm "${DECOMPOSER_LLM}" \
  --subtask-source "${SUBTASK_SOURCE}" \
  --result-dir "result/exp4_workflow_func_subtask/${STUDENT_MODEL}"
