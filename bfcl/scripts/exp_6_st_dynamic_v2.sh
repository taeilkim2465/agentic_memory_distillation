#!/usr/bin/env bash
# Experiment 6: Workflow (static, Turn 1) + Subtask (system_replace, per-turn) + Function
#
# Workflow memory : retrieved once before Turn 1, stays in system prompt throughout.
# Subtask memory  : retrieved at each turn using that turn's user instruction,
#                   replaces the subtask section of the system prompt (not user message).
# Function memory : appended to error execution results at each turn.
#
# Differs from exp5 (dynamic): injection goes into system prompt, not user message,
# so FC model behavior is not disrupted.
set -euo pipefail

STUDENT_MODEL="qwen3-4b-FC-vllm"
TEST_CATEGORY="multi_turn_base"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8001/v1}"
MEMORY_DATE="0516"
NUM_THREADS=10
TEMPERATURE=0.0
RESULT_TAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --student-model) STUDENT_MODEL="$2"; shift 2 ;;
    --vllm-url)      VLLM_BASE_URL="$2"; shift 2 ;;
    --test-category) TEST_CATEGORY="$2"; shift 2 ;;
    --memory-date)   MEMORY_DATE="$2";   shift 2 ;;
    --num-threads)   NUM_THREADS="$2";   shift 2 ;;
    --temperature)   TEMPERATURE="$2";   shift 2 ;;
    --result-tag)    RESULT_TAG="$2";    shift 2 ;;
    -*)  echo "[ERROR] Unknown option: $1"; exit 1 ;;
    *)   shift ;;
  esac
done

PYTHON="${BFCL_PYTHON:-python3}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MEMORY_DIR="${REPO_ROOT}/data/memory/${MEMORY_DATE}"

# result dir: model name에 tag와 날짜시간 suffix 추가
DATETIME_TAG="$(date +%m%d_%H%M)"
if [[ -n "${RESULT_TAG}" ]]; then
  RESULT_SUBDIR="${STUDENT_MODEL}_${RESULT_TAG}_${DATETIME_TAG}"
else
  RESULT_SUBDIR="${STUDENT_MODEL}"
fi

echo "========================================"
echo " Exp 6: Workflow(static) + Subtask(system_replace/per-turn) + Function"
echo "  Model      : ${STUDENT_MODEL}"
echo "  vLLM URL   : ${VLLM_BASE_URL}"
echo "  Category   : ${TEST_CATEGORY}"
echo "  Memory dir : ${MEMORY_DIR}"
echo "  Threads    : ${NUM_THREADS}"
echo "  Result tag : ${RESULT_TAG:-（없음）}"
echo "  Result dir : result/exp6_st_dynamic_v2/${RESULT_SUBDIR}"
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
  --memory-types "workflow,subtask,function" \
  --memory-retrieval "system_replace" \
  --result-dir "result/exp6_st_dynamic_v2/${RESULT_SUBDIR}"
