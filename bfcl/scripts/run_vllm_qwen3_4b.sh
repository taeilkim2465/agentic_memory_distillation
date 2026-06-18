#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-1}"
PORT="${PORT:-8221}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-4b}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.98}"

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export HF_HOME="${HF_HOME:-/c2/taeil/huggingface}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/c2/taeil/.cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/c2/taeil/.cache/torch/inductor}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/c2/taeil/.cache/vllm}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_ID}" \
exec vllm serve "$MODEL" \
  --port "$PORT" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
