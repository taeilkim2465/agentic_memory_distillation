#!/usr/bin/env bash
# MEMP Teacher: GPT-5-Mini로 시나리오 실행 후 trajectory를 절차화(proceduralize)하여
# MEMP memory store에 저장한다. 실패 시 기존 메모리를 in-place 수정(Adjustment).
#
# 사용:
#   bash run_memp_teacher.sh [옵션]
#
# 옵션:
#   --memp_store   PATH   store 파일 경로      (기본: memp-memories/store.json)
#   --keyword_llm  MODEL  키워드 추출 LLM      (기본: gpt-5-mini)
#   --proc_llm     MODEL  절차화 LLM           (기본: gpt-5-mini)
#   --adjust_llm   MODEL  조정 LLM             (기본: gpt-5-mini)
#   -p, --parallel N      병렬 프로세스 수      (기본: 16)
#   -o, --output   DIR    출력 디렉토리         (기본: data)
set -euo pipefail

# ── 기본값 ────────────────────────────────────────────────────────────────────
MEMP_STORE="${MEMP_STORE:-memp-memories/store.json}"
KEYWORD_LLM="${KEYWORD_LLM:-gpt-5-mini}"
PROC_LLM="${PROC_LLM:-gpt-5-mini}"
ADJUST_LLM="${ADJUST_LLM:-gpt-5-mini}"
PARALLEL="${PARALLEL:-16}"
OUTPUT_DIR="${OUTPUT_DIR:-data}"
PYTHON="${PYTHON:-/c2/taeil/anaconda3/envs/ToolSandbox/bin/tool_sandbox}"
PYBIN="${PYBIN:-/c2/taeil/anaconda3/envs/ToolSandbox/bin/python}"

# ── 인수 파싱 ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --memp_store)   MEMP_STORE="$2";   shift 2 ;;
    --keyword_llm)  KEYWORD_LLM="$2";  shift 2 ;;
    --proc_llm)     PROC_LLM="$2";     shift 2 ;;
    --adjust_llm)   ADJUST_LLM="$2";   shift 2 ;;
    -p|--parallel)  PARALLEL="$2";     shift 2 ;;
    -o|--output)    OUTPUT_DIR="$2";   shift 2 ;;
    *) echo "알 수 없는 옵션: $1"; exit 1 ;;
  esac
done

# ── base 129 시나리오 (NO_DISTRACTION_TOOLS) ──────────────────────────────────
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

echo "=== MEMP Teacher: GPT_5_Mini agent + GPT_5_Mini user ==="
echo "MEMP_STORE  : $MEMP_STORE"
echo "KEYWORD_LLM : $KEYWORD_LLM"
echo "PROC_LLM    : $PROC_LLM"
echo "ADJUST_LLM  : $ADJUST_LLM"
echo "PARALLEL    : $PARALLEL"
echo "OUTPUT_DIR  : $OUTPUT_DIR"
echo ""

mkdir -p "$(dirname "$MEMP_STORE")"

cd /c2/taeil/ToolSandbox && "$PYTHON" \
  --agent GPT_5_Mini \
  --user GPT_5_Mini \
  --build_memp \
  --memp_store_path "$MEMP_STORE" \
  --memp_keyword_llm "$KEYWORD_LLM" \
  --memp_proceduralize_llm "$PROC_LLM" \
  --memp_adjust_llm "$ADJUST_LLM" \
  --similarity_threshold 1.0 \
  -p "$PARALLEL" \
  -o "$OUTPUT_DIR" \
  --scenarios $BASE_SCENARIOS

echo ""
echo "=== MEMP Teacher complete ==="
if [[ -f "$MEMP_STORE" ]]; then
  "$PYBIN" -c "
import json
with open('$MEMP_STORE') as f:
    store = json.load(f)
s = sum(1 for e in store if e.get('success'))
fa = sum(1 for e in store if not e.get('success'))
print(f'Store: {len(store)}개 entry  (성공:{s} / 실패:{fa})')
print(f'경로: $MEMP_STORE')
" 2>/dev/null || true
fi
echo ""
echo "다음 단계: bash run_memp_student.sh --memp_store $MEMP_STORE"
