#!/usr/bin/env bash
set -euo pipefail

# Student experiments using teacher memories from teacher_GPT_5_Mini_user_GPT_5_Mini_0519_1927
# Run after starting vLLM: bash run_vllm_qwen3_4b.sh

VLLM_URL="${VLLM_URL:-http://localhost:8221/v1}"
MEMORY_DIR="${MEMORY_DIR:-memories/teacher_GPT_5_Mini_user_GPT_5_Mini_0519_1927}"
PARALLEL="${PARALLEL:-16}"
OUTPUT_DIR="${OUTPUT_DIR:-data}"
PYTHON="${PYTHON:-/c2/taeil/anaconda3/envs/ToolSandbox/bin/tool_sandbox}"

echo "=== Student runs: teacher_GPT_5_Mini_user_GPT_5_Mini_0519_1927 ==="
echo "VLLM_URL  : $VLLM_URL"
echo "MEMORY_DIR: $MEMORY_DIR"
echo "PARALLEL  : $PARALLEL"
echo ""

# 1. Baseline (no memory)
echo "[1/4] Qwen3_4B (no memory)"
cd /c2/taeil/ToolSandbox && "$PYTHON" \
  --agent Qwen3_4B \
  --user GPT_5_Mini \
  --vllm-url "$VLLM_URL" \
  -p "$PARALLEL" \
  -o "$OUTPUT_DIR"
echo "[1/4] Done"

# 2. WF only
echo "[2/4] MemoryQwen3_4B_WF"
cd /c2/taeil/ToolSandbox && "$PYTHON" \
  --agent MemoryQwen3_4B_WF \
  --user GPT_5_Mini \
  --vllm-url "$VLLM_URL" \
  --memory_dir "$MEMORY_DIR" \
  -p "$PARALLEL" \
  -o "$OUTPUT_DIR"
echo "[2/4] Done"

# 3. WF + FN
echo "[3/4] MemoryQwen3_4B_WF_FN"
cd /c2/taeil/ToolSandbox && "$PYTHON" \
  --agent MemoryQwen3_4B_WF_FN \
  --user GPT_5_Mini \
  --vllm-url "$VLLM_URL" \
  --memory_dir "$MEMORY_DIR" \
  -p "$PARALLEL" \
  -o "$OUTPUT_DIR"
echo "[3/4] Done"

# 4. WF + FN + ST
echo "[4/4] MemoryQwen3_4B_WF_FN_ST"
cd /c2/taeil/ToolSandbox && "$PYTHON" \
  --agent MemoryQwen3_4B_WF_FN_ST \
  --user GPT_5_Mini \
  --vllm-url "$VLLM_URL" \
  --memory_dir "$MEMORY_DIR" \
  -p "$PARALLEL" \
  -o "$OUTPUT_DIR"
echo "[4/4] Done"

echo ""
echo "=== All student runs complete ==="
echo "Summarizing results..."
/c2/taeil/anaconda3/envs/ToolSandbox/bin/python \
  /c2/taeil/ToolSandbox/scripts/summarize_results.py \
  /c2/taeil/ToolSandbox/data/student_*_user_GPT_5_Mini_* 2>/dev/null || true
