#!/bin/bash
# RB Student — qwen3-4b via vLLM
#
# Usage:
#   bash rb_run_student_qwen3_4b.sh [options]
#
# Options:
#   --vllm-url URL       vLLM endpoint (default: http://10.10.0.118:8001/v1)
#   --num-processes N    parallel workers (default: 1)
#   --task-id ID         run a single task

set -e
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

VLLM_URL="http://10.10.0.118:8001/v1"
NUM_PROCESSES=""
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --vllm-url)      VLLM_URL="$2";      shift 2 ;;
        --num-processes) NUM_PROCESSES="$2";  shift 2 ;;
        *)               PASSTHROUGH+=("$1"); shift   ;;
    esac
done

OVERRIDE=$(python -c "import json; print(json.dumps({'config': {'agent': {'generator_model_config': {'base_url': '$VLLM_URL'}}}}))")

CMD=(bash "$BASE_DIR/rb_run.sh" ACE_offline_no_GT_evaluation_qwen3_4b_student --override "$OVERRIDE")
[ -n "$NUM_PROCESSES" ] && CMD+=(--num-processes "$NUM_PROCESSES")
CMD+=("${PASSTHROUGH[@]}")

"${CMD[@]}"
