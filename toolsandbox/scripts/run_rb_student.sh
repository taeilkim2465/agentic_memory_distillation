#!/usr/bin/env bash
# RB Student: teacher가 만든 bank를 읽어 top-k memory를 주입한 뒤 시나리오를 실행한다.
# baseline(메모리 없음)과 RB(메모리 주입)를 순서대로 실행해 비교한다.
#
# 사용:
#   bash run_rb_student.sh [옵션]
#
# 옵션:
#   --model          MODEL  실행할 모델            (기본: qwen3-4b)
#                           qwen3-4b | qwen3-8b | gemma4-e4b | llama3.1-8b
#   --rb_bank        PATH   bank 파일 경로         (기본: rb-memories/bank.json)
#   --vllm_url       URL    vLLM 엔드포인트        (기본: http://localhost:8221/v1)
#   --top_k          N      검색할 메모리 개수      (기본: 3)
#   --skip_baseline        baseline 실행 건너뜀    (이미 결과 있을 때)
#   --all_scenarios        전체 시나리오 실행       (기본: base 129개만)
#   -p, --parallel   N      병렬 프로세스 수        (기본: 16)
#   -o, --output     DIR    출력 디렉토리           (기본: data)
set -euo pipefail

# ── 기본값 ────────────────────────────────────────────────────────────────────
MODEL="${MODEL:-qwen3-4b}"
RB_BANK="${RB_BANK:-rb-memories/bank.json}"
VLLM_URL="${VLLM_URL:-http://localhost:8221/v1}"
RB_TOP_K="${RB_TOP_K:-3}"
SKIP_BASELINE=1
ALL_SCENARIOS=0
PARALLEL="${PARALLEL:-16}"
OUTPUT_DIR="${OUTPUT_DIR:-data}"
PYTHON="${PYTHON:-/c2/taeil/anaconda3/envs/ToolSandbox/bin/tool_sandbox}"
PYBIN="${PYBIN:-/c2/taeil/anaconda3/envs/ToolSandbox/bin/python}"

# ── 인수 파싱 ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)           MODEL="$2";        shift 2 ;;
    --rb_bank)         RB_BANK="$2";      shift 2 ;;
    --vllm_url)        VLLM_URL="$2";     shift 2 ;;
    --top_k)           RB_TOP_K="$2";     shift 2 ;;
    --skip_baseline)   SKIP_BASELINE=1;   shift   ;;
    --all_scenarios)   ALL_SCENARIOS=1;   shift   ;;
    -p|--parallel)     PARALLEL="$2";     shift 2 ;;
    -o|--output)       OUTPUT_DIR="$2";   shift 2 ;;
    *) echo "알 수 없는 옵션: $1"; exit 1 ;;
  esac
done

# ── 모델 → agent type 매핑 ────────────────────────────────────────────────────
case "$MODEL" in
  qwen3-4b)    BASELINE_AGENT="Qwen3_4B";   RB_AGENT="RBQwen3_4B"   ;;
  qwen3-8b)    BASELINE_AGENT="Qwen3_8B";   RB_AGENT="RBQwen3_8B"   ;;
  gemma4-e4b)  BASELINE_AGENT="Gemma4E4B";  RB_AGENT="RBGemma4E4B"  ;;
  llama3.1-8b) BASELINE_AGENT="Llama31_8B"; RB_AGENT="RBLlama31_8B" ;;
  *)
    echo "지원 모델: qwen3-4b | qwen3-8b | gemma4-e4b | llama3.1-8b"
    exit 1
    ;;
esac

# ── bank 파일 확인 ────────────────────────────────────────────────────────────
if [[ ! -f "$RB_BANK" ]]; then
  echo "bank 파일 없음: $RB_BANK"
  echo "먼저 실행: bash run_rb_teacher.sh --rb_bank $RB_BANK"
  exit 1
fi

ENTRY_COUNT=$("$PYBIN" -c "import json; print(len(json.load(open('$RB_BANK'))))" 2>/dev/null || echo "?")

echo "=== RB Student: $MODEL / $RB_AGENT + GPT_5_Mini user ==="
echo "VLLM_URL      : $VLLM_URL"
echo "RB_BANK       : $RB_BANK  ($ENTRY_COUNT entries)"
echo "RB_TOP_K      : $RB_TOP_K"
echo "PARALLEL      : $PARALLEL"
echo "OUTPUT_DIR    : $OUTPUT_DIR"
echo "ALL_SCENARIOS : $ALL_SCENARIOS"
echo ""

cd /c2/taeil/ToolSandbox

# ── 시나리오 필터 ─────────────────────────────────────────────────────────────
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
"
  )
  SCENARIO_FILTER="--scenarios $BASE_SCENARIOS"
fi

# ── [1/2] Baseline — 메모리 없음 ─────────────────────────────────────────────
if [[ "$SKIP_BASELINE" -eq 1 ]]; then
  echo "[1/2] Baseline SKIP (--skip_baseline)"
else
  echo "[1/2] Baseline: $BASELINE_AGENT"
  "$PYTHON" \
    --agent "$BASELINE_AGENT" \
    --user GPT_5_Mini \
    --vllm-url "$VLLM_URL" \
    -p "$PARALLEL" \
    -o "$OUTPUT_DIR" \
    $SCENARIO_FILTER
  echo "[1/2] 완료"
fi
echo ""

# ── [2/2] RB Student — Reasoning Bank 메모리 주입 ────────────────────────────
echo "[2/2] RB: $RB_AGENT (top-$RB_TOP_K)"
"$PYTHON" \
  --agent "$RB_AGENT" \
  --user GPT_5_Mini \
  --vllm-url "$VLLM_URL" \
  --rb_bank_path "$RB_BANK" \
  --rb_top_k "$RB_TOP_K" \
  -p "$PARALLEL" \
  -o "$OUTPUT_DIR" \
  $SCENARIO_FILTER
echo "[2/2] 완료"
echo ""

# ── 결과 요약 ─────────────────────────────────────────────────────────────────
echo "=== 결과 요약 ==="
"$PYBIN" \
  /c2/taeil/ToolSandbox/scripts/summarize_results.py \
  "$OUTPUT_DIR"/agent_"${BASELINE_AGENT}"_user_GPT_5_Mini_* \
  "$OUTPUT_DIR"/rb_student_"${RB_AGENT}"_user_GPT_5_Mini_* \
  2>/dev/null || true
