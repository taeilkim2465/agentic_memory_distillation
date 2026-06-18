#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

bash memp-memories/run_student.sh \
  --model         qwen3-8b-FC-vllm \
  --vllm-url      http://localhost:8661 \
  --num-threads   8 \
  --result-dir    result/memory_student/memp/qwen3-8b-FC-vllm \
  --evaluate \
  "$@"
