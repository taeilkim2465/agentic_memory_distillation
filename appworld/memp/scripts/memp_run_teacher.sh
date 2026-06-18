#!/bin/bash
# Mem^p Teacher — builds memory from test_normal
# Usage: bash memp_run_teacher.sh [--task-id <id>] [--num-processes N]

set -e
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$BASE_DIR/memp"

APPWORLD_ROOT="$REPO_DIR" PYTHONPATH="$REPO_DIR:$PYTHONPATH" appworld run MEMP_teacher_no_GT "$@"
