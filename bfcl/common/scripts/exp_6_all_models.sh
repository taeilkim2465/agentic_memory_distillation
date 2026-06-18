#!/usr/bin/env bash
# Run Experiment 6 (system_replace) for all four models sequentially.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP6="${SCRIPT_DIR}/exp_6_st_dynamic_v2.sh"

MODELS=(
  "qwen3-4b-FC-vllm"
  "qwen3-8b-FC-vllm"
  "gemma4-e4b-FC-vllm"
  "llama3.1-8b-FC-vllm"
)

# Pass-through args (e.g. --memory-date, --test-category, --num-threads)
EXTRA_ARGS=("$@")

for MODEL in "${MODELS[@]}"; do
  echo ""
  echo "###############################################"
  echo "  Running Exp 6 — model: ${MODEL}"
  echo "###############################################"
  bash "${EXP6}" --student-model "${MODEL}" "${EXTRA_ARGS[@]}"
done

echo ""
echo "All models done."
