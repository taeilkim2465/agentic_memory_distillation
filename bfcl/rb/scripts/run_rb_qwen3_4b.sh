#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

bash rb-memories/run_eval.sh \
  --model         qwen3-4b-FC-vllm \
  --student-url   http://localhost:8881 \
  --num-threads   4 \
  --result-dir    result/memory_student/rb/qwen3-4b-FC-vllm \
  --evaluate \
  "$@"
