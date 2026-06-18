#!/usr/bin/env bash
# Gemma4-E4B baseline: 메모리 없이 base 129개(NO_DISTRACTION_TOOLS) 시나리오 실행
#
# 사용:
#   bash run_student_gemma4_e4b_baseline.sh [옵션]
#
# 옵션:
#   --vllm_url  URL   vLLM 엔드포인트  (기본: http://localhost:8881/v1)
#   --all_scenarios   전체 시나리오 실행 (기본: base 129개만)
#   -p, --parallel N  병렬 프로세스 수  (기본: 16)
#   -o, --output   DIR 출력 디렉토리   (기본: data)
set -euo pipefail

VLLM_URL="${VLLM_URL:-http://localhost:8881/v1}"
ALL_SCENARIOS=0
PARALLEL="${PARALLEL:-16}"
OUTPUT_DIR="${OUTPUT_DIR:-data}"
PYTHON="${PYTHON:-tool_sandbox}"
PYBIN="${PYBIN:-python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --vllm_url|--vllm-url) VLLM_URL="$2";   shift 2 ;;
    --all_scenarios)        ALL_SCENARIOS=1; shift   ;;
    -p|--parallel)          PARALLEL="$2";   shift 2 ;;
    -o|--output|--output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "알 수 없는 옵션: $1"; exit 1 ;;
  esac
done

echo "=== Gemma4E4B Baseline ==="
echo "VLLM_URL      : $VLLM_URL"
echo "PARALLEL      : $PARALLEL"
echo "OUTPUT_DIR    : $OUTPUT_DIR"
echo "ALL_SCENARIOS : $ALL_SCENARIOS"
echo ""

cd "${TOOLSANDBOX_ROOT:?Set TOOLSANDBOX_ROOT to your ToolSandbox repo path}"

if [[ "$ALL_SCENARIOS" -eq 1 ]]; then
  SCENARIO_FILTER=""
else
  BASE_SCENARIOS=$(
    "$PYBIN" -c "
from tool_sandbox.scenarios import named_scenarios
from tool_sandbox.common.tool_discovery import ToolBackend
from tool_sandbox.common.execution_context import ScenarioCategories
scenarios = named_scenarios(preferred_tool_backend=ToolBackend.DEFAULT)
print(' '.join(sorted(
    n for n, s in scenarios.items()
    if ScenarioCategories.NO_DISTRACTION_TOOLS in s.categories
)))
" 2>/dev/null
  )
  SCENARIO_FILTER="--scenarios $BASE_SCENARIOS"
fi

"$PYTHON" \
  --agent Gemma4E4B \
  --user GPT_5_Mini \
  --vllm-url "$VLLM_URL" \
  -p "$PARALLEL" \
  -o "$OUTPUT_DIR" \
  $SCENARIO_FILTER

echo "완료"
