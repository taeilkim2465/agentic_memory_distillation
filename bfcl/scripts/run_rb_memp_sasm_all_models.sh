#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Student inference: rb / memp / sasm  ×  4 models (sequential)
#
# Models and vLLM endpoints:
#   qwen3-4b-FC-vllm    localhost:8881
#   qwen3-8b-FC-vllm    localhost:8661
#   llama3.1-8b-FC-vllm localhost:8331
#   gemma4-e4b-FC-vllm  localhost:8991
#
# Usage:
#   bash run/run_rb_memp_sasm_all_models.sh [options]
#
# Options:
#   --test-category  CATEGORY   multi_turn_base
#   --num-threads    N          16
#   --top-k          N          3
#   --result-dir     PATH       result/memory_student
#   --allow-overwrite           (flag)
#   --no-evaluate               skip bfcl evaluate after each run (default: evaluate)
#
# Note on result directories:
#   rb   → <result-dir>/rb/<model>/<timestamp>/          (fixed path, evaluated separately)
#   memp → <result-dir>/memp/<model>/<internal-timestamp>/  (memp creates timestamp internally)
#   sasm → <result-dir>/sasm/<model>/<internal-timestamp>/  (sasm creates timestamp internally)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

# ── Defaults ─────────────────────────────────────────────────────────────────
TEST_CATEGORY="multi_turn_base"
NUM_THREADS=16
TOP_K=3
BASE_RESULT_DIR="result/memory_student"
ALLOW_OVERWRITE=""
DO_EVALUATE="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --test-category)   TEST_CATEGORY="$2";    shift 2 ;;
    --num-threads)     NUM_THREADS="$2";      shift 2 ;;
    --top-k)           TOP_K="$2";            shift 2 ;;
    --result-dir)      BASE_RESULT_DIR="$2";  shift 2 ;;
    --allow-overwrite) ALLOW_OVERWRITE="--allow-overwrite"; shift ;;
    --no-evaluate)     DO_EVALUATE="";        shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

declare -A MODEL_URLS=(
  ["qwen3-4b-FC-vllm"]="http://localhost:8881"
  ["qwen3-8b-FC-vllm"]="http://localhost:8661"
  ["llama3.1-8b-FC-vllm"]="http://localhost:8331"
  ["gemma4-e4b-FC-vllm"]="http://localhost:8991"
)
MODELS=("qwen3-4b-FC-vllm" "qwen3-8b-FC-vllm" "llama3.1-8b-FC-vllm" "gemma4-e4b-FC-vllm")

EVALUATE_FLAG=""
[[ -n "$DO_EVALUATE" ]] && EVALUATE_FLAG="--evaluate"

section() {
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo " $*"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

for MODEL in "${MODELS[@]}"; do
  URL="${MODEL_URLS[$MODEL]}"

  # ── RB ────────────────────────────────────────────────────────────────────
  # run_eval.sh internally appends its own timestamp subdir → pass --evaluate to handle it inside
  section "[RB] model=${MODEL}  url=${URL}"
  bash rb-memories/run_eval.sh \
    --model         "$MODEL" \
    --test-category "$TEST_CATEGORY" \
    --rb-bank-dir   "rb-memories/bank" \
    --top-k         "$TOP_K" \
    --num-threads   "$NUM_THREADS" \
    --student-url   "$URL" \
    --result-dir    "${BASE_RESULT_DIR}/rb/${MODEL}" \
    ${ALLOW_OVERWRITE} \
    ${EVALUATE_FLAG}

  # ── MEMP ──────────────────────────────────────────────────────────────────
  # memp internally appends its own timestamp subdir → pass --evaluate to handle it inside
  section "[MEMP] model=${MODEL}  url=${URL}"
  bash memp-memories/run_student.sh \
    --model          "$MODEL" \
    --test-category  "$TEST_CATEGORY" \
    --memp-store-dir "memp-memories/store" \
    --memp-llm-model "gpt-5-mini" \
    --top-k          "$TOP_K" \
    --num-threads    "$NUM_THREADS" \
    --student-url    "$URL" \
    --result-dir     "${BASE_RESULT_DIR}/memp/${MODEL}" \
    ${ALLOW_OVERWRITE} \
    ${EVALUATE_FLAG}

  # ── SASM ──────────────────────────────────────────────────────────────────
  # sasm internally appends its own timestamp subdir → pass --evaluate to handle it inside
  section "[SASM] model=${MODEL}  url=${URL}"
  bash sasm-memories/run_student.sh \
    --model          "$MODEL" \
    --test-category  "$TEST_CATEGORY" \
    --sasm-store-dir "sasm-memories/store" \
    --sasm-llm-model "gpt-5-mini" \
    --num-threads    "$NUM_THREADS" \
    --student-url    "$URL" \
    --result-dir     "${BASE_RESULT_DIR}/sasm/${MODEL}" \
    ${ALLOW_OVERWRITE} \
    ${EVALUATE_FLAG}

done

section "All done.  category=${TEST_CATEGORY}"
echo " RB results  : ${BASE_RESULT_DIR}/rb/<model>/<timestamp>/"
echo " MEMP results: ${BASE_RESULT_DIR}/memp/<model>/<timestamp>/"
echo " SASM results: ${BASE_RESULT_DIR}/sasm/<model>/<timestamp>/"
