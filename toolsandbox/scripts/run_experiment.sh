#!/usr/bin/env bash
# 전체 실험 파이프라인: 메모리 빌드 → RB student → MEMP student → 결과 비교
#
# 사용:
#   bash run_experiment.sh [옵션]
#
# 옵션:
#   --model       MODEL   테스트 모델          (기본: qwen3-4b)
#                          qwen3-4b | qwen3-8b | gemma4-e4b | llama3.1-8b
#   --vllm_url    URL     vLLM 엔드포인트      (기본: http://localhost:8221/v1)
#   --rb_bank     PATH    RB bank 파일 경로    (기본: rb-memories/bank.json)
#   --memp_store  PATH    MEMP store 파일 경로 (기본: memp-memories/store.json)
#   --run_dir     DIR     teacher 실행 디렉토리 (메모리 빌드 시 필요)
#   --top_k       N       검색 메모리 수        (기본: 3)
#   --rebuild            메모리 강제 재빌드
#   -p, --parallel N     병렬 프로세스 수      (기본: 16)
#   -o, --output  DIR    출력 디렉토리         (기본: data)
set -euo pipefail

# ── 기본값 ────────────────────────────────────────────────────────────────────
MODEL="${MODEL:-qwen3-4b}"
VLLM_URL="${VLLM_URL:-http://localhost:8221/v1}"
RB_BANK="${RB_BANK:-rb-memories/bank.json}"
MEMP_STORE="${MEMP_STORE:-memp-memories/store.json}"
RUN_DIR="${RUN_DIR:-}"
TOP_K="${TOP_K:-3}"
REBUILD=0
PARALLEL="${PARALLEL:-16}"
OUTPUT_DIR="${OUTPUT_DIR:-data}"
PYTHON="${PYTHON:-/c2/taeil/anaconda3/envs/ToolSandbox/bin/tool_sandbox}"
PYBIN="${PYBIN:-/c2/taeil/anaconda3/envs/ToolSandbox/bin/python}"

# ── 인수 파싱 ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)       MODEL="$2";      shift 2 ;;
    --vllm_url)    VLLM_URL="$2";   shift 2 ;;
    --rb_bank)     RB_BANK="$2";    shift 2 ;;
    --memp_store)  MEMP_STORE="$2"; shift 2 ;;
    --run_dir)     RUN_DIR="$2";    shift 2 ;;
    --top_k)       TOP_K="$2";      shift 2 ;;
    --rebuild)     REBUILD=1;       shift   ;;
    -p|--parallel) PARALLEL="$2";   shift 2 ;;
    -o|--output)   OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "알 수 없는 옵션: $1"; exit 1 ;;
  esac
done

# ── 모델 → agent type 매핑 ────────────────────────────────────────────────────
case "$MODEL" in
  qwen3-4b)    BASELINE_AGENT="Qwen3_4B";   RB_AGENT="RBQwen3_4B";   MEMP_AGENT="MEMPQwen3_4B"   ;;
  qwen3-8b)    BASELINE_AGENT="Qwen3_8B";   RB_AGENT="RBQwen3_8B";   MEMP_AGENT="MEMPQwen3_8B"   ;;
  gemma4-e4b)  BASELINE_AGENT="Gemma4E4B";  RB_AGENT="RBGemma4E4B";  MEMP_AGENT="MEMPGemma4E4B"  ;;
  llama3.1-8b) BASELINE_AGENT="Llama31_8B"; RB_AGENT="RBLlama31_8B"; MEMP_AGENT="MEMPLlama31_8B" ;;
  *)
    echo "지원 모델: qwen3-4b | qwen3-8b | gemma4-e4b | llama3.1-8b"
    exit 1
    ;;
esac

cd /c2/taeil/ToolSandbox

echo "================================================================"
echo " 실험 파이프라인: $MODEL"
echo "================================================================"
echo "VLLM_URL   : $VLLM_URL"
echo "RB_BANK    : $RB_BANK"
echo "MEMP_STORE : $MEMP_STORE"
echo "TOP_K      : $TOP_K"
echo "PARALLEL   : $PARALLEL"
echo "OUTPUT_DIR : $OUTPUT_DIR"
echo ""

# ── NO_DISTRACTION_TOOLS 129개 시나리오 ──────────────────────────────────────
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

# ── [1/4] Teacher 실행 (메모리가 없거나 --rebuild 시) ─────────────────────────
NEED_BUILD=0
[[ "$REBUILD" -eq 1 ]]                          && NEED_BUILD=1
[[ ! -f "$RB_BANK" || ! -f "$MEMP_STORE" ]]     && NEED_BUILD=1

if [[ "$NEED_BUILD" -eq 1 ]]; then
  if [[ -z "$RUN_DIR" ]]; then
    echo "━━━ [1/4] Teacher Baseline 실행 (GPT_5_Mini, no memory) ━━━"
    "$PYTHON" \
      --agent GPT_5_Mini \
      --user GPT_5_Mini \
      -p "$PARALLEL" \
      -o "$OUTPUT_DIR" \
      --scenarios $BASE_SCENARIOS

    RUN_DIR=$(ls -dt "$OUTPUT_DIR"/agent_GPT_5_Mini_user_GPT_5_Mini_* 2>/dev/null | head -1)
    [[ -z "$RUN_DIR" ]] && { echo "teacher 실행 디렉토리를 찾을 수 없음"; exit 1; }
    echo "[1/4] 완료 → $RUN_DIR"
  else
    echo "━━━ [1/4] Teacher Baseline SKIP (--run_dir $RUN_DIR) ━━━"
  fi

  # ── [2/4] RB bank 빌드 ──────────────────────────────────────────────────────
  if [[ ! -f "$RB_BANK" || "$REBUILD" -eq 1 ]]; then
    echo ""
    echo "━━━ [2/4] RB Bank 빌드 ━━━"
    mkdir -p "$(dirname "$RB_BANK")"
    "$PYBIN" scripts/build_rb_from_existing.py \
      --run_dir "$RUN_DIR" \
      --rb_bank "$RB_BANK" \
      --similarity_threshold 1.0 \
      --parallel "$PARALLEL"
    echo "[2/4] 완료 → $RB_BANK"
  else
    echo "━━━ [2/4] RB Bank SKIP (이미 존재) ━━━"
  fi

  # ── [3/4] MEMP store 빌드 ───────────────────────────────────────────────────
  if [[ ! -f "$MEMP_STORE" || "$REBUILD" -eq 1 ]]; then
    echo ""
    echo "━━━ [3/4] MEMP Store 빌드 ━━━"
    mkdir -p "$(dirname "$MEMP_STORE")"
    "$PYBIN" scripts/build_memp_from_existing.py \
      --run_dir "$RUN_DIR" \
      --memp_store "$MEMP_STORE" \
      --similarity_threshold 1.0 \
      --parallel "$PARALLEL"
    echo "[3/4] 완료 → $MEMP_STORE"
  else
    echo "━━━ [3/4] MEMP Store SKIP (이미 존재) ━━━"
  fi
else
  echo "━━━ [1-3/4] 메모리 빌드 SKIP (이미 존재, --rebuild로 강제 재빌드) ━━━"
  RB_COUNT=$("$PYBIN" -c "import json; print(len(json.load(open('$RB_BANK'))))" 2>/dev/null || echo "?")
  MEMP_COUNT=$("$PYBIN" -c "import json; print(len(json.load(open('$MEMP_STORE'))))" 2>/dev/null || echo "?")
  echo "  RB Bank    : $RB_BANK  ($RB_COUNT entries)"
  echo "  MEMP Store : $MEMP_STORE  ($MEMP_COUNT entries)"
fi

echo ""

# ── [4/4a] RB Student 실행 ───────────────────────────────────────────────────
echo "━━━ [4a/4] RB Student: $RB_AGENT (top-$TOP_K) ━━━"
"$PYTHON" \
  --agent "$RB_AGENT" \
  --user GPT_5_Mini \
  --vllm-url "$VLLM_URL" \
  --rb_bank_path "$RB_BANK" \
  --rb_top_k "$TOP_K" \
  -p "$PARALLEL" \
  -o "$OUTPUT_DIR" \
  --scenarios $BASE_SCENARIOS
echo "[4a/4] RB 완료"
echo ""

# ── [4/4b] MEMP Student 실행 ─────────────────────────────────────────────────
echo "━━━ [4b/4] MEMP Student: $MEMP_AGENT (top-$TOP_K) ━━━"
"$PYTHON" \
  --agent "$MEMP_AGENT" \
  --user GPT_5_Mini \
  --vllm-url "$VLLM_URL" \
  --memp_store_path "$MEMP_STORE" \
  --memp_top_k "$TOP_K" \
  -p "$PARALLEL" \
  -o "$OUTPUT_DIR" \
  --scenarios $BASE_SCENARIOS
echo "[4b/4] MEMP 완료"
echo ""

# ── 결과 요약 ─────────────────────────────────────────────────────────────────
echo "================================================================"
echo " 결과 요약"
echo "================================================================"
"$PYBIN" -c "
import json, glob, os

def load_sim(pattern):
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        return None, None
    d = dirs[-1]  # 가장 최신 실행
    p = os.path.join(d, 'result_summary.json')
    if not os.path.exists(p):
        return None, None
    with open(p) as f:
        data = json.load(f)
    sims = [r['similarity'] for r in data['per_scenario_results']]
    return sims, os.path.basename(d)

rb_sims,   rb_dir   = load_sim('$OUTPUT_DIR/rb_student_${RB_AGENT}_user_GPT_5_Mini_*')
memp_sims, memp_dir = load_sim('$OUTPUT_DIR/memp_student_${MEMP_AGENT}_user_GPT_5_Mini_*')

def fmt(sims, label, dirname):
    if sims is None:
        print(f'  {label}: 결과 없음')
        return
    mean = sum(sims)/len(sims)
    full = sum(1 for s in sims if s >= 1.0)
    print(f'  {label} ({dirname}):')
    print(f'    n={len(sims)}, mean={mean:.4f}, sim=1.0: {full}개 ({full/len(sims)*100:.1f}%)')

fmt(rb_sims,   'RB  ', rb_dir)
fmt(memp_sims, 'MEMP', memp_dir)

if rb_sims and memp_sims:
    delta = sum(memp_sims)/len(memp_sims) - sum(rb_sims)/len(rb_sims)
    print(f'  Δ MEMP - RB = {delta:+.4f}')
" 2>/dev/null || true
