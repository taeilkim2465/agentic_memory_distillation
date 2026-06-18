#!/usr/bin/env bash
# RB Teacher: GPT-5-Mini로 시나리오 실행 후 성공/실패 trajectory 모두에서
# Reasoning Bank memory item을 추출해 bank 파일에 저장한다.
#
# 사용:
#   bash run_rb_teacher.sh [옵션]
#
# 옵션:
#   --rb_bank      PATH   bank 파일 경로       (기본: rb-memories/bank.json)
#   --reflector    MODEL  메모리 생성 LLM      (기본: gpt-5-mini)
#   -p, --parallel N      병렬 프로세스 수      (기본: 16)
#   -o, --output   DIR    출력 디렉토리         (기본: data)
set -euo pipefail

# ── 기본값 ────────────────────────────────────────────────────────────────────
RB_BANK="${RB_BANK:-rb-memories/bank.json}"
RB_REFLECTOR_LLM="${RB_REFLECTOR_LLM:-gpt-5-mini}"
PARALLEL="${PARALLEL:-16}"
OUTPUT_DIR="${OUTPUT_DIR:-data}"
PYTHON="${PYTHON:-/c2/taeil/anaconda3/envs/ToolSandbox/bin/tool_sandbox}"
PYBIN="${PYBIN:-/c2/taeil/anaconda3/envs/ToolSandbox/bin/python}"

# ── 인수 파싱 ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --rb_bank)      RB_BANK="$2";          shift 2 ;;
    --reflector)    RB_REFLECTOR_LLM="$2"; shift 2 ;;
    -p|--parallel)  PARALLEL="$2";         shift 2 ;;
    -o|--output)    OUTPUT_DIR="$2";       shift 2 ;;
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

echo "=== RB Teacher: GPT_5_Mini agent + GPT_5_Mini user ==="
echo "RB_BANK         : $RB_BANK"
echo "RB_REFLECTOR_LLM: $RB_REFLECTOR_LLM"
echo "PARALLEL        : $PARALLEL"
echo "OUTPUT_DIR      : $OUTPUT_DIR"
echo "저장 대상       : 성공 + 실패 trajectory 모두"
echo ""

mkdir -p "$(dirname "$RB_BANK")"

cd /c2/taeil/ToolSandbox && "$PYTHON" \
  --agent GPT_5_Mini \
  --user GPT_5_Mini \
  --build_rb \
  --rb_bank_path "$RB_BANK" \
  --rb_reflector_llm "$RB_REFLECTOR_LLM" \
  --similarity_threshold 1.0 \
  -p "$PARALLEL" \
  -o "$OUTPUT_DIR" \
  --scenarios $BASE_SCENARIOS

echo ""
echo "=== RB Teacher complete ==="
if [[ -f "$RB_BANK" ]]; then
  "$PYBIN" -c "
import json
with open('$RB_BANK') as f:
    bank = json.load(f)
s = sum(1 for e in bank if e.get('outcome') == 'success')
f = sum(1 for e in bank if e.get('outcome') == 'failure')
items = sum(len(e.get('memory_items', [])) for e in bank)
print(f'Bank: {len(bank)}개 entry / {items}개 memory item  (성공:{s} / 실패:{f})')
print(f'경로: $RB_BANK')
" 2>/dev/null || true
fi
echo ""
echo "다음 단계: bash run_rb_student.sh --rb_bank $RB_BANK"
