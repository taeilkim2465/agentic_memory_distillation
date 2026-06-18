#!/usr/bin/env bash
set -euo pipefail

VLLM_URL="${VLLM_URL:-http://localhost:8881/v1}"
MEMORY_DIR="${MEMORY_DIR:-memories/teacher_GPT_5_Mini_user_GPT_5_Mini_0519_1927}"
PARALLEL="${PARALLEL:-16}"
OUTPUT_DIR="${OUTPUT_DIR:-data}"
PYTHON="${PYTHON:-tool_sandbox}"

echo "=== Student runs (4 variants in parallel) ==="
echo "VLLM_URL  : $VLLM_URL"
echo "MEMORY_DIR: $MEMORY_DIR"
echo "PARALLEL  : $PARALLEL"
echo ""

cd "${TOOLSANDBOX_ROOT:?Set TOOLSANDBOX_ROOT to your ToolSandbox repo path}"

# 1. Baseline (no memory)
echo "[1/4] Qwen3_4B baseline 시작..."
"$PYTHON" \
  --agent Qwen3_4B \
  --user GPT_5_Mini \
  --vllm-url "$VLLM_URL" \
  -p "$PARALLEL" \
  -o "$OUTPUT_DIR"
echo "[1/4] 완료"

# 2. WF only
echo "[2/4] MemoryQwen3_4B_WF 시작..."
"$PYTHON" \
  --agent MemoryQwen3_4B_WF \
  --user GPT_5_Mini \
  --vllm-url "$VLLM_URL" \
  --memory_dir "$MEMORY_DIR" \
  -p "$PARALLEL" \
  -o "$OUTPUT_DIR"
echo "[2/4] 완료"

# 3. WF + FN
echo "[3/4] MemoryQwen3_4B_WF_FN 시작..."
"$PYTHON" \
  --agent MemoryQwen3_4B_WF_FN \
  --user GPT_5_Mini \
  --vllm-url "$VLLM_URL" \
  --memory_dir "$MEMORY_DIR" \
  -p "$PARALLEL" \
  -o "$OUTPUT_DIR"
echo "[3/4] 완료"

# 4. WF + FN + ST
echo "[4/4] MemoryQwen3_4B_WF_FN_ST 시작..."
"$PYTHON" \
  --agent MemoryQwen3_4B_WF_FN_ST \
  --user GPT_5_Mini \
  --vllm-url "$VLLM_URL" \
  --memory_dir "$MEMORY_DIR" \
  -p "$PARALLEL" \
  -o "$OUTPUT_DIR"
echo "[4/4] 완료"

echo ""
echo "=== 전체 완료 ==="
python \
  ${TOOLSANDBOX_ROOT}/scripts/summarize_results.py \
  ${TOOLSANDBOX_ROOT}/data/student_*_user_GPT_5_Mini_* 2>/dev/null || true
