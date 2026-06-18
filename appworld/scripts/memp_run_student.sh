#!/bin/bash
# Mem^p Student — solves test_normal using teacher's memory
# Usage: bash memp_run_student.sh [--task-id <id>] [--num-processes N]
#
# Prerequisite: memp_run_teacher.sh must have completed first.

set -e
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$BASE_DIR/memp"
MEMORY_FILE="$REPO_DIR/experiments/memory/memp_teacher_store.json"

if [ ! -f "$MEMORY_FILE" ]; then
    echo "ERROR: Teacher memory not found at $MEMORY_FILE"
    echo "       Run memp_run_teacher.sh first."
    exit 1
fi

ENTRY_COUNT=$(python -c "import json; d=json.load(open('$MEMORY_FILE')); print(len(d))")
echo "Teacher memory loaded: $ENTRY_COUNT entries"

APPWORLD_ROOT="$REPO_DIR" PYTHONPATH="$REPO_DIR:$PYTHONPATH" appworld run MEMP_student_no_GT "$@"
