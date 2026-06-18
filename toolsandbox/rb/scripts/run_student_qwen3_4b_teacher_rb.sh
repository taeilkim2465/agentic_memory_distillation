#!/usr/bin/env bash
# Qwen3-4B student: teacher memory (WF/WF_FN/WF_FN_ST) + RB memory 순차 실행
set -euo pipefail

VLLM_URL="${VLLM_URL:-http://localhost:8221/v1}"
MEMORY_DIR="${MEMORY_DIR:-memories/teacher_GPT_5_Mini_user_GPT_5_Mini_0520_1210}"
RB_BANK="${RB_BANK:-rb-memories/bank.json}"
RB_TOP_K="${RB_TOP_K:-3}"
PARALLEL="${PARALLEL:-16}"
OUTPUT_DIR="${OUTPUT_DIR:-data}"
PYTHON="${PYTHON:-tool_sandbox}"
PYBIN="${PYBIN:-python}"

echo "=== Qwen3-4B Student: Teacher + RB Memory ==="
echo "VLLM_URL  : $VLLM_URL"
echo "MEMORY_DIR: $MEMORY_DIR"
echo "RB_BANK   : $RB_BANK  ($("$PYBIN" -c "import json; print(len(json.load(open('$RB_BANK'))))" 2>/dev/null || echo "?") entries)"
echo "RB_TOP_K  : $RB_TOP_K"
echo "PARALLEL  : $PARALLEL"
echo ""

cd "${TOOLSANDBOX_ROOT:?Set TOOLSANDBOX_ROOT to your ToolSandbox repo path}"

# ── [1/3] Teacher WF ──────────────────────────────────────────────────────────
echo "[1/3] MemoryQwen3_4B_WF"
"$PYTHON" \
  --agent MemoryQwen3_4B_WF \
  --user GPT_5_Mini \
  --vllm-url "$VLLM_URL" \
  --memory_dir "$MEMORY_DIR" \
  -p "$PARALLEL" \
  -o "$OUTPUT_DIR"
echo "[1/3] 완료"
echo ""

# ── [2/3] Teacher WF + FN + ST ────────────────────────────────────────────────
echo "[2/3] MemoryQwen3_4B_WF_FN_ST"
"$PYTHON" \
  --agent MemoryQwen3_4B_WF_FN_ST \
  --user GPT_5_Mini \
  --vllm-url "$VLLM_URL" \
  --memory_dir "$MEMORY_DIR" \
  -p "$PARALLEL" \
  -o "$OUTPUT_DIR"
echo "[2/3] 완료"
echo ""

# ── [3/3] RB Student ──────────────────────────────────────────────────────────
echo "[3/3] RBQwen3_4B (top-$RB_TOP_K)"
"$PYTHON" \
  --agent RBQwen3_4B \
  --user GPT_5_Mini \
  --vllm-url "$VLLM_URL" \
  --rb_bank_path "$RB_BANK" \
  --rb_top_k "$RB_TOP_K" \
  -p "$PARALLEL" \
  -o "$OUTPUT_DIR"
echo "[3/3] 완료"
echo ""

# ── 결과 요약 ─────────────────────────────────────────────────────────────────
echo "=== 결과 요약 ==="
"$PYBIN" \
  ${TOOLSANDBOX_ROOT}/scripts/summarize_results.py \
  "$OUTPUT_DIR"/student_MemoryQwen3_4B_WF_user_GPT_5_Mini_* \
  "$OUTPUT_DIR"/student_MemoryQwen3_4B_WF_FN_ST_user_GPT_5_Mini_* \
  "$OUTPUT_DIR"/rb_student_RBQwen3_4B_user_GPT_5_Mini_* \
  2>/dev/null || true
