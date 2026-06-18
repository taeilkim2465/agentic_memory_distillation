#!/bin/bash
# Reasoning Bank (RB) — run any experiment config
# Usage: bash rb_run.sh <experiment_name> [--task-id <id>] [--num-processes N]
#
# Example: bash rb_run.sh ACE_offline_no_GT_adaptation_gpt5mini_teacher
#          bash rb_run.sh ACE_offline_no_GT_evaluation_qwen3_4b_student

set -e
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$BASE_DIR/rb"

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <experiment_name> [appworld run options...]"
    echo "Available configs:"
    ls "$REPO_DIR/experiments/configs/"*.jsonnet 2>/dev/null | xargs -n1 basename | sed 's/\.jsonnet//'
    exit 1
fi

EXPERIMENT_NAME="$1"
shift

APPWORLD_PROJECT_PATH="$REPO_DIR" APPWORLD_ROOT="$REPO_DIR" PYTHONPATH="$REPO_DIR:$PYTHONPATH" \
    appworld run "$EXPERIMENT_NAME" "$@"
