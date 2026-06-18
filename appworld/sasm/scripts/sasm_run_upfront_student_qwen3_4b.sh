#!/bin/bash
# SASM Upfront Student — qwen3-4b via vLLM
#
# Decomposes task instruction upfront using student LLM,
# classifies predicted subtasks into z categories,
# retrieves matching experiences from memory bank,
# then runs normal ReAct loop with all experiences pre-injected.
#
# Usage:
#   bash sasm_run_upfront_student_qwen3_4b.sh [options]
#
# Options:
#   --vllm-url URL       vLLM endpoint (default: http://localhost:8881/v1)
#   --num-processes N    parallel workers (default: 1)
#   --memory-path PATH   SASM memory file (default: sasm/experiments/playbooks/sasm_memory.json)
#   --task-id ID         run a single task
#
# Prerequisite: sasm_run_teacher.sh must have completed first.

set -e
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$BASE_DIR/sasm"
DEFAULT_MEMORY="$REPO_DIR/experiments/playbooks/sasm_memory.json"

if [ -z "$OPENAI_API_KEY" ]; then
    source ~/.bashrc
fi
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY is not set."
    exit 1
fi

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
    echo "ERROR: SASM memory not found at $MEMORY_FILE"
    echo "       Run sasm_run_teacher.sh first, or pass --memory-path."
    exit 1
fi

ENTRY_COUNT=$(python -c "import json; d=json.load(open('$MEMORY_FILE')); print(len(d.get('entries', d)))")
echo "SASM memory: $MEMORY_FILE ($ENTRY_COUNT entries)"

CMD=(appworld run SASM_upfront_evaluation_qwen3_4b_student)

[ -n "$VLLM_URL" ]      && CMD+=(--vllm-url "$VLLM_URL")
[ -n "$NUM_PROCESSES" ] && CMD+=(--num-processes "$NUM_PROCESSES")
CMD+=("${PASSTHROUGH[@]}")

# Override memory path when non-default requested
if [ -n "$MEMORY_PATH" ]; then
    OVERRIDE=$(python -c "import json; print(json.dumps({'config': {'agent': {'sasm_memory_file_path': '$MEMORY_PATH'}}}))")
    CMD+=(--override "$OVERRIDE")
fi

APPWORLD_PROJECT_PATH="$REPO_DIR" APPWORLD_ROOT="$REPO_DIR" PYTHONPATH="$REPO_DIR:$PYTHONPATH" "${CMD[@]}"
