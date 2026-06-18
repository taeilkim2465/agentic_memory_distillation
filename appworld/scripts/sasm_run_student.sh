#!/bin/bash
# SASM Student — evaluates on test_normal using teacher-built memory
# Usage: bash sasm_run_student.sh [--task-id <id>] [--num-processes N]
#
# Prerequisite: sasm_run_teacher.sh must have completed first.
# Config: sasm/experiments/configs/SASM_evaluation.jsonnet
# Memory: sasm/experiments/playbooks/sasm_memory.json

set -e
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$BASE_DIR/sasm"

if [ -z "$OPENAI_API_KEY" ]; then
    source ~/.bashrc
fi
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY is not set."
    exit 1
fi

MEMORY_FILE="$REPO_DIR/experiments/playbooks/sasm_memory.json"

if [ ! -f "$MEMORY_FILE" ]; then
    echo "ERROR: SASM memory not found at $MEMORY_FILE"
    echo "       Run sasm_run_teacher.sh first."
    exit 1
fi

ENTRY_COUNT=$(python -c "import json; d=json.load(open('$MEMORY_FILE')); print(len(d))")
echo "SASM memory loaded: $ENTRY_COUNT entries"

APPWORLD_PROJECT_PATH="$REPO_DIR" APPWORLD_ROOT="$REPO_DIR" PYTHONPATH="$REPO_DIR:$PYTHONPATH" \
    appworld run SASM_evaluation "$@"
