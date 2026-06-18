#!/bin/bash
# SASM Teacher — solves tasks and builds sasm_memory.json
# Usage: bash sasm_run_teacher.sh [--task-id <id>] [--num-processes N] [--sample-size N]
#
# Config: sasm/experiments/configs/SASM_adaptation.jsonnet
# Output: sasm/experiments/playbooks/sasm_memory.json

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

APPWORLD_PROJECT_PATH="$REPO_DIR" APPWORLD_ROOT="$REPO_DIR" PYTHONPATH="$REPO_DIR:$PYTHONPATH" \
    appworld run SASM_adaptation "$@"
