#!/usr/bin/env bash
# student_wf_subtask_func_multi_turn.sh
#
# Run qwen3-4b-FC on BFCL multi_turn_base with all three memory types:
#   - Workflow memory  → injected into system prompt (similar past tasks)
#   - Subtask memory   → injected into system prompt (tool call examples)
#   - Function memory  → injected dynamically when tool calls fail
#
# Requires: QWEN_API_KEY env var set
# Memory must be pre-built by teacher_build_memory_multi_turn.sh first.
#
# Usage:
#   bash run/student_wf_subtask_func_multi_turn.sh
#   bash run/student_wf_subtask_func_multi_turn.sh --memory-date 0516
#   bash run/student_wf_subtask_func_multi_turn.sh --num-threads 20 --memory-date 0516

set -euo pipefail

STUDENT_MODEL="gemma4-e4b-FC-vllm"
DECOMPOSER_LLM="gpt-4o-mini"
TEST_CATEGORY="multi_turn_base"
MEMORY_DATE="$(date +%m%d)"
NUM_THREADS=10      # Qwen hosted API: start conservatively, raise if rate limits allow
TEMPERATURE=0.0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --student-model)   STUDENT_MODEL="$2";    shift 2 ;;
    --decomposer-llm)  DECOMPOSER_LLM="$2";   shift 2 ;;
    --test-category)   TEST_CATEGORY="$2";    shift 2 ;;
    --memory-date)     MEMORY_DATE="$2";      shift 2 ;;
    --num-threads)     NUM_THREADS="$2";      shift 2 ;;
    --temperature)     TEMPERATURE="$2";      shift 2 ;;
    -*)  echo "[ERROR] Unknown option: $1"; exit 1 ;;
    *)   shift ;;
  esac
done

PYTHON="${BFCL_PYTHON:-python3}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MEMORY_DIR="${REPO_ROOT}/data/memory/${MEMORY_DATE}"

echo "========================================"
echo " BFCL Student (workflow + subtask + function memory)"
echo "  Student model  : ${STUDENT_MODEL}"
echo "  Decomposer LLM : ${DECOMPOSER_LLM}"
echo "  Test category  : ${TEST_CATEGORY}"
echo "  Memory dir     : ${MEMORY_DIR}"
echo "  Threads        : ${NUM_THREADS}"
echo "========================================"

# API key / endpoint check (model-aware)
if [[ "${STUDENT_MODEL}" == *"-vllm" ]]; then
  VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8001/v1}"
  echo "[INFO] vLLM endpoint: ${VLLM_BASE_URL}"
elif [[ "${STUDENT_MODEL}" == *"qwen"* ]] || [[ "${STUDENT_MODEL}" == *"Qwen"* ]]; then
  if [ -z "${QWEN_API_KEY:-}" ]; then
    echo "[ERROR] QWEN_API_KEY is not set."
    exit 1
  fi
fi

# Check memory files exist
for F in "${MEMORY_DIR}/workflow/documents.json" \
         "${MEMORY_DIR}/subtask/segments.jsonl" \
         "${MEMORY_DIR}/function/records.jsonl"; do
  if [ ! -f "${F}" ]; then
    echo "[ERROR] Memory file not found: ${F}"
    echo "        Run teacher_build_memory_multi_turn.sh first."
    exit 1
  fi
done

echo ""
echo "⏳ Running student inference with memory..."
cd "${REPO_ROOT}"

"${PYTHON}" -m bfcl_eval generate \
  --model "${STUDENT_MODEL}" \
  --test-category "${TEST_CATEGORY}" \
  --temperature "${TEMPERATURE}" \
  --num-threads "${NUM_THREADS}" \
  --memory-dir "${MEMORY_DIR}" \
  --decomposer-llm "${DECOMPOSER_LLM}"

LATEST_RESULT_DIR="$(ls -dt "${REPO_ROOT}/result/"*/ 2>/dev/null | head -1)"

echo ""
echo "========================================"
echo " ✅ Done. Results in: ${LATEST_RESULT_DIR}"
echo ""
echo " Run evaluation with:"
echo "   python -m bfcl_eval evaluate \\"
echo "     --model ${STUDENT_MODEL} \\"
echo "     --test-category ${TEST_CATEGORY} \\"
echo "     --result-dir ${LATEST_RESULT_DIR}"
echo "========================================"
