#!/usr/bin/env bash
# SASM teacher phase: run GPT-5-Mini agent on base 129 scenarios,
# decompose trajectories into subtasks, extract experience entries.
set -euo pipefail

PARALLEL="${PARALLEL:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-data}"
SASM_STORE="${SASM_STORE:-sasm-memories/store.json}"
PREDICTOR_LLM="${PREDICTOR_LLM:-gpt-5-mini}"
PYTHON="${PYTHON:-/c2/taeil/anaconda3/envs/ToolSandbox/bin/tool_sandbox}"
VLLM_URL="${VLLM_URL:-}"

BASE_SCENARIOS=$(
  /c2/taeil/anaconda3/envs/ToolSandbox/bin/python -c "
from tool_sandbox.scenarios import named_scenarios
from tool_sandbox.common.tool_discovery import ToolBackend
from tool_sandbox.common.execution_context import ScenarioCategories
scenarios = named_scenarios(preferred_tool_backend=ToolBackend.DEFAULT)
base = sorted(
    name for name, sc in scenarios.items()
    if ScenarioCategories.NO_DISTRACTION_TOOLS in sc.categories
)
print(' '.join(base))
"
)

echo "=== SASM Teacher: GPT_5_Mini + base 129 scenarios ==="
echo "PARALLEL   : $PARALLEL"
echo "SASM_STORE : $SASM_STORE"
echo ""

mkdir -p "$(dirname "$SASM_STORE")"

VLLM_ARG=""
if [[ -n "$VLLM_URL" ]]; then
    VLLM_ARG="--vllm-url $VLLM_URL"
fi

cd /c2/taeil/ToolSandbox && "$PYTHON" \
  --agent GPT_5_Mini \
  --user GPT_5_Mini \
  --build_sasm \
  --sasm_store_path "$SASM_STORE" \
  --sasm_predictor_llm "$PREDICTOR_LLM" \
  --similarity_threshold 1.0 \
  -p "$PARALLEL" \
  -o "$OUTPUT_DIR" \
  $VLLM_ARG \
  --scenarios $BASE_SCENARIOS

echo ""
echo "=== SASM Teacher complete ==="
echo "Store: $SASM_STORE"
