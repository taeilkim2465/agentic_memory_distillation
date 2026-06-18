#!/usr/bin/env bash
# run_vllm_gemma4_e4b.sh
#
# Gemma4-e4b vLLM 서버 시작
# gemma4_exp_*.sh 실행 전에 먼저 이 스크립트를 실행해야 합니다.
#
# Usage:
#   bash run/run_vllm_gemma4_e4b.sh
#   GPU_ID=2 PORT=8002 bash run/run_vllm_gemma4_e4b.sh
#
# 선택 파라미터:
#   GPU_ID - GPU 번호 (기본값: 1)
#   PORT   - 서버 포트 (기본값: 8001)
#
# 서버가 준비됐는지 확인:
#   curl http://localhost:8001/v1/models

set -euo pipefail

MODEL="google/gemma-4-e4b-it"
GPU_ID="${GPU_ID:-1}"
PORT="${PORT:-8001}"
SERVED_MODEL_NAME="gemma4-e4b"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
TENSOR_PARALLEL="${VLLM_TP:-1}"

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export HF_HOME="${HF_HOME:-/c2/taeil/huggingface}"
export HF_HUB_DISABLE_XET=1
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/c2/taeil/.cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/c2/taeil/.cache/torch/inductor}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/c2/taeil/.cache/vllm}"

echo "========================================"
echo " Starting vLLM server (Gemma4-e4b)"
echo "  Model  : ${MODEL}"
echo "  Served : ${SERVED_MODEL_NAME}"
echo "  Port   : ${PORT}"
echo "  GPU ID : ${GPU_ID}"
echo "  GPU mem: ${GPU_MEMORY_UTILIZATION}"
echo "========================================"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_ID}" \
exec vllm serve "${MODEL}" \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --tensor-parallel-size "${TENSOR_PARALLEL}" \
  --enable-auto-tool-choice \
  --tool-call-parser pythonic
