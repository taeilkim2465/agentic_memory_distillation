#!/bin/bash
# Mem^p Student — qwen3-4b via vLLM (teacher: gpt-4o-mini)
#
# Usage:
#   bash memp_run_student_qwen3_4b.sh [options]
#
# Options:
#   --vllm-url URL       vLLM endpoint (default: http://10.10.0.118:8001/v1)
#   --num-processes N    parallel workers (default: 1)
#   --memory-path PATH   teacher memory file (default: memp/experiments/memory/memp_teacher_store.json)
#   --task-id ID         run a single task
#   --tag TAG            output directory tag
#
# Prerequisite: memp_run_teacher.sh must have completed first.

set -e
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$BASE_DIR/memp"
DEFAULT_MEMORY="$REPO_DIR/experiments/memory/memp_teacher_store.json"

VLLM_URL=""
NUM_PROCESSES=""
MEMORY_PATH=""
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --vllm-url)      VLLM_URL="$2";      shift 2 ;;
        --num-processes) NUM_PROCESSES="$2";  shift 2 ;;
        --memory-path)   MEMORY_PATH="$2";   shift 2 ;;
        *)               PASSTHROUGH+=("$1"); shift   ;;
    esac
done

MEMORY_FILE="${MEMORY_PATH:-$DEFAULT_MEMORY}"

if [ ! -f "$MEMORY_FILE" ]; then
    echo "ERROR: Teacher memory not found at $MEMORY_FILE"
    echo "       Run memp_run_teacher.sh first, or pass --memory-path."
    exit 1
fi

ENTRY_COUNT=$(python -c "import json; d=json.load(open('$MEMORY_FILE')); print(len(d))")
echo "Teacher memory: $MEMORY_FILE ($ENTRY_COUNT entries)"

CMD=(appworld run MEMP_student_no_GT_qwen3_4b)

[ -n "$VLLM_URL" ]      && CMD+=(--vllm-url "$VLLM_URL")
[ -n "$NUM_PROCESSES" ] && CMD+=(--num-processes "$NUM_PROCESSES")
CMD+=("${PASSTHROUGH[@]}")

# Override memory path only when non-default is requested
if [ -n "$MEMORY_PATH" ]; then
    OVERRIDE=$(python -c "import json; print(json.dumps({'config': {'agent': {'memory_store_path': '$MEMORY_PATH'}}}))")
    CMD+=(--override "$OVERRIDE")
fi

APPWORLD_ROOT="$REPO_DIR" PYTHONPATH="$REPO_DIR:$PYTHONPATH" "${CMD[@]}"
