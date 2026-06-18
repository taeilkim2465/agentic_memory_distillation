#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

bash rb-memories/run_eval.sh \
  --model         gemma4-e4b-FC-vllm \
  --student-url   http://localhost:8991 \
  --num-threads   8 \
  --result-dir    result/memory_student/rb/gemma4-e4b-FC-vllm \
  --evaluate \
  "$@"
