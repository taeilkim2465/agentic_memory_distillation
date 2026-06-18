#!/bin/bash
# SASM Full Pipeline — teacher builds memory, then student evaluates
# Usage: bash sasm_run_full.sh [--num-processes N]
#
# Phase 1 (teacher): solves train tasks, writes sasm/experiments/playbooks/sasm_memory.json
# Phase 2 (student): evaluates on test_normal using that memory
# Config: sasm/experiments/configs/SASM_full.jsonnet

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
    appworld run SASM_full "$@"
