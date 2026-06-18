#!/usr/bin/env bash
# teacher_build_memory_multi_turn.sh
#
# Step 1: Run a strong teacher model on BFCL multi_turn_base test cases.
# Step 2: Extract workflow, subtask, and function memories from the teacher trajectories.
#
# Usage:
#   bash run/teacher_build_memory_multi_turn.sh
#   bash run/teacher_build_memory_multi_turn.sh --teacher-model gpt-5-mini-2025-08-07-FC
#   bash run/teacher_build_memory_multi_turn.sh --memory-date 0516

set -euo pipefail

TEACHER_MODEL="gpt-5-mini-2025-08-07-FC"
TEACHER_LLM="gpt-5-mini-2025-08-07"  # litellm model string for memory builder (workflow/subtask extraction)
THINK_MODEL=""                         # if set, generate think annotations during teacher inference
EMBEDDING_MODEL="text-embedding-3-small"
TEST_CATEGORY="multi_turn_base"
MEMORY_DATE="$(date +%m%d)"
RESULT_DIR=""    # auto-resolved after generate
SCORE_FILE=""    # optional: filter to successful tasks only

while [[ $# -gt 0 ]]; do
  case "$1" in
    --teacher-model)   TEACHER_MODEL="$2";    shift 2 ;;
    --teacher-llm)     TEACHER_LLM="$2";      shift 2 ;;
    --think-model)     THINK_MODEL="$2";      shift 2 ;;
    --test-category)   TEST_CATEGORY="$2";    shift 2 ;;
    --memory-date)     MEMORY_DATE="$2";      shift 2 ;;
    --result-dir)      RESULT_DIR="$2";       shift 2 ;;
    --score-file)      SCORE_FILE="$2";       shift 2 ;;
    -*)  echo "[ERROR] Unknown option: $1"; exit 1 ;;
    *)   shift ;;
  esac
done

PYTHON="${BFCL_PYTHON:-python3}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MEMORY_DIR="${REPO_ROOT}/data/memory/${MEMORY_DATE}"

echo "========================================"
echo " BFCL Teacher Memory Builder"
echo "  Teacher model : ${TEACHER_MODEL}"
echo "  Think model   : ${THINK_MODEL:-'(disabled)'}"
echo "  Test category : ${TEST_CATEGORY}"
echo "  Memory dir    : ${MEMORY_DIR}"
echo "  Score file    : ${SCORE_FILE:-'(none — all tasks included)'}"
echo "========================================"

# ─── Step 1: Run teacher inference ───────────────────────────────────────────
echo ""
echo "⏳ Step 1: Running teacher inference on ${TEST_CATEGORY}..."
cd "${REPO_ROOT}"

THINK_ARGS=()
if [[ -n "${THINK_MODEL}" ]]; then
  THINK_ARGS=(--think-model "${THINK_MODEL}")
fi

"${PYTHON}" -m bfcl_eval generate \
  --model "${TEACHER_MODEL}" \
  --test-category "${TEST_CATEGORY}" \
  "${THINK_ARGS[@]}"

# Find the latest result directory
LATEST_RESULT_DIR="$(ls -dt "${REPO_ROOT}/result/"*/ 2>/dev/null | head -1)"
if [ -z "${RESULT_DIR}" ]; then
  RESULT_DIR="${LATEST_RESULT_DIR}"
fi

echo "  Result dir: ${RESULT_DIR}"

# Collect result files for this test category
TEACHER_SLUG="${TEACHER_MODEL//\//_}"
RESULT_FILES=()
while IFS= read -r -d '' f; do
  RESULT_FILES+=("$f")
done < <(find "${RESULT_DIR}/${TEACHER_SLUG}" -name "*${TEST_CATEGORY}*result.json" -print0 2>/dev/null)

if [ ${#RESULT_FILES[@]} -eq 0 ]; then
  echo "[ERROR] No result files found in ${RESULT_DIR}/${TEACHER_SLUG}"
  exit 1
fi

echo "  Found ${#RESULT_FILES[@]} result file(s): ${RESULT_FILES[*]}"

# ─── Step 2: Build memory ─────────────────────────────────────────────────────
echo ""
echo "⏳ Step 2: Building memory from teacher trajectories..."

SCORE_ARGS=()
if [[ -n "${SCORE_FILE}" ]]; then
  SCORE_ARGS=(--score-file "${SCORE_FILE}")
fi

"${PYTHON}" "${REPO_ROOT}/run/build_memory_from_results.py" \
  --result-files "${RESULT_FILES[@]}" \
  --memory-dir "${MEMORY_DIR}" \
  --teacher-llm "${TEACHER_LLM}" \
  --embedding-model "${EMBEDDING_MODEL}" \
  --test-category "${TEST_CATEGORY}" \
  "${SCORE_ARGS[@]}"

echo ""
echo "========================================"
echo " ✅ Done. Memory files:"
ls -lh "${MEMORY_DIR}/workflow/" "${MEMORY_DIR}/subtask/" "${MEMORY_DIR}/function/" 2>/dev/null || true
echo " Memory dir: ${MEMORY_DIR}"
echo "========================================"
