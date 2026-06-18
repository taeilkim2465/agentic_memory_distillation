#!/usr/bin/env bash
# Evaluate all 4 experiments for a given timestamp tag.
#
# Usage:
#   bash run/evaluate_all.sh 0516_2014
#   bash run/evaluate_all.sh 0516_2014 --exps 3,4        # only exp3, exp4
#   bash run/evaluate_all.sh 0516_2014 --model qwen3-4b-FC-vllm
#   bash run/evaluate_all.sh 0516_2014 --prefix gemma4_  # gemma4_ 접두사 실험

set -euo pipefail

TIMESTAMP="${1:-}"
if [[ -z "$TIMESTAMP" ]]; then
  echo "[ERROR] 타임스탬프를 첫 번째 인자로 전달해주세요."
  echo "  Usage: bash run/evaluate_all.sh 0516_2014"
  exit 1
fi
shift

MODEL="${DEFAULT_MODEL:-qwen3-4b-FC-vllm}"
TEST_CATEGORY="multi_turn_base"
EXPS="1,2,3,4"
PREFIX="${EXP_PREFIX:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)    MODEL="$2";    shift 2 ;;
    --category) TEST_CATEGORY="$2"; shift 2 ;;
    --exps)     EXPS="$2";     shift 2 ;;
    --prefix)   PREFIX="$2";   shift 2 ;;
    *) echo "[ERROR] Unknown option: $1"; exit 1 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${BFCL_PYTHON:-conda run -n bfcl python}"

declare -A EXP_DIRS
EXP_DIRS[1]="${PREFIX}exp1_baseline"
EXP_DIRS[2]="${PREFIX}exp2_workflow"
EXP_DIRS[3]="${PREFIX}exp3_workflow_func"
EXP_DIRS[4]="${PREFIX}exp4_workflow_func_subtask"

echo "========================================"
echo " Evaluate experiments"
echo "  Timestamp : ${TIMESTAMP}"
echo "  Model     : ${MODEL}"
echo "  Category  : ${TEST_CATEGORY}"
echo "  Exps      : ${EXPS}"
echo "========================================"

IFS=',' read -ra EXP_LIST <<< "$EXPS"
FAILED=()

for EXP_NUM in "${EXP_LIST[@]}"; do
  EXP_LABEL="${EXP_DIRS[$EXP_NUM]:-}"
  if [[ -z "$EXP_LABEL" ]]; then
    echo "[WARN] 알 수 없는 실험 번호: $EXP_NUM — 건너뜀"
    continue
  fi

  RESULT_DIR="${REPO_ROOT}/result/${EXP_LABEL}/${MODEL}/${TIMESTAMP}"
  SCORE_DIR="${REPO_ROOT}/score/${EXP_LABEL}"

  if [[ ! -d "$RESULT_DIR" ]]; then
    echo "[SKIP] exp${EXP_NUM} — 결과 없음: ${RESULT_DIR}"
    continue
  fi

  echo ""
  echo "--- exp${EXP_NUM}: ${EXP_LABEL} ---"
  output=$($PYTHON -m bfcl_eval evaluate \
      --model "$MODEL" \
      --test-category "$TEST_CATEGORY" \
      --result-dir "result/${EXP_LABEL}/${MODEL}/${TIMESTAMP}" \
      --score-dir "score/${EXP_LABEL}" 2>&1 || true)
  echo "$output" | grep -E "Accuracy|KeyError|SKIP|completed" || true
  if echo "$output" | grep -q "Accuracy"; then
    echo "  → score saved: ${SCORE_DIR}"
  else
    echo "  [WARN] exp${EXP_NUM} 평가 실패"
    FAILED+=("exp${EXP_NUM}")
  fi
done

echo ""
echo "========================================"
echo " 결과 요약"
echo "========================================"
python "${REPO_ROOT}/run/summarize_run.py" "$TIMESTAMP"

if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo ""
  echo "[WARN] 실패한 실험: ${FAILED[*]}"
  exit 1
fi
