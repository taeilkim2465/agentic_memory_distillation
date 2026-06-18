#!/usr/bin/env bash
# teacher_build_memory_v3_extra.sh
#
# v3 나머지 4개 카테고리(miss_func, miss_param, long_context, composite)에 대해
# teacher inference를 실행하고 기존 메모리에 append한다.
#
# Usage:
#   bash run/teacher_build_memory_v3_extra.sh
#   bash run/teacher_build_memory_v3_extra.sh --memory-date 0518 --think-model gpt-5-mini-2025-08-07

set -euo pipefail

TEACHER_MODEL="gpt-5-mini-2025-08-07-FC"
TEACHER_LLM="gpt-5-mini-2025-08-07"
THINK_MODEL=""
EMBEDDING_MODEL="text-embedding-3-small"
MEMORY_DATE="0518"
SCORE_FILE=""

V3_CATEGORIES=(
    "multi_turn_miss_func"
    "multi_turn_miss_param"
    "multi_turn_long_context"
    "multi_turn_composite"
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --teacher-model)   TEACHER_MODEL="$2";    shift 2 ;;
    --teacher-llm)     TEACHER_LLM="$2";      shift 2 ;;
    --think-model)     THINK_MODEL="$2";      shift 2 ;;
    --memory-date)     MEMORY_DATE="$2";      shift 2 ;;
    --score-file)      SCORE_FILE="$2";       shift 2 ;;
    -*)  echo "[ERROR] Unknown option: $1"; exit 1 ;;
    *)   shift ;;
  esac
done

PYTHON="${BFCL_PYTHON:-python3}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MEMORY_DIR="${REPO_ROOT}/data/memory/${MEMORY_DATE}"

echo "========================================"
echo " BFCL Teacher Memory Builder (v3 extra)"
echo "  Teacher model : ${TEACHER_MODEL}"
echo "  Think model   : ${THINK_MODEL:-'(disabled)'}"
echo "  Memory dir    : ${MEMORY_DIR} (append mode)"
echo "  Categories    : ${V3_CATEGORIES[*]}"
echo "========================================"

THINK_ARGS=()
if [[ -n "${THINK_MODEL}" ]]; then
  THINK_ARGS=(--think-model "${THINK_MODEL}")
fi

SCORE_ARGS=()
if [[ -n "${SCORE_FILE}" ]]; then
  SCORE_ARGS=(--score-file "${SCORE_FILE}")
fi

TEACHER_SLUG="${TEACHER_MODEL//\//_}"
ALL_RESULT_FILES=()

# ─── Step 1: 각 카테고리 teacher inference ───────────────────────────────────
for CATEGORY in "${V3_CATEGORIES[@]}"; do
  echo ""
  echo "⏳ Running teacher inference: ${CATEGORY}..."
  cd "${REPO_ROOT}"

  "${PYTHON}" -m bfcl_eval generate \
    --model "${TEACHER_MODEL}" \
    --test-category "${CATEGORY}" \
    "${THINK_ARGS[@]}"

  # 가장 최근 result 디렉토리에서 해당 카테고리 파일 탐색
  LATEST_DIR="$(ls -dt "${REPO_ROOT}/result/"*/ 2>/dev/null | head -1)"
  while IFS= read -r -d '' f; do
    ALL_RESULT_FILES+=("$f")
    echo "  Found: $f"
  done < <(find "${LATEST_DIR}/${TEACHER_SLUG}" -name "*${CATEGORY}*result.json" -print0 2>/dev/null)
done

if [ ${#ALL_RESULT_FILES[@]} -eq 0 ]; then
  echo "[ERROR] No result files found."
  exit 1
fi

echo ""
echo "⏳ Step 2: Building memory (append to ${MEMORY_DIR})..."
echo "  Result files: ${#ALL_RESULT_FILES[@]} files"

"${PYTHON}" "${REPO_ROOT}/run/build_memory_from_results.py" \
  --result-files "${ALL_RESULT_FILES[@]}" \
  --memory-dir "${MEMORY_DIR}" \
  --teacher-llm "${TEACHER_LLM}" \
  --embedding-model "${EMBEDDING_MODEL}" \
  --test-category "${V3_CATEGORIES[@]}" \
  "${SCORE_ARGS[@]}"

echo ""
echo "========================================"
echo " ✅ Done. Memory files:"
ls -lh "${MEMORY_DIR}/workflow/" "${MEMORY_DIR}/subtask/" "${MEMORY_DIR}/function/" 2>/dev/null || true
echo " Memory dir: ${MEMORY_DIR}"
echo "========================================"
