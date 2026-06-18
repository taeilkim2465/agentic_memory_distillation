#!/usr/bin/env bash
# 기존 teacher trajectory에서 시나리오 재실행 없이 SASM 메모리만 빌드
#
# 사용:
#   bash run_sasm_from_existing.sh [옵션]
#
# 옵션:
#   --teacher_dir   DIR   기존 teacher 실행 디렉토리 (기본: data/teacher_GPT_5_Mini_user_GPT_5_Mini_0520_1133)
#   --sasm_store    PATH  출력 store 파일 경로       (기본: sasm-memories/store.json)
#   --predictor_llm LLM  키워드 추출 LLM            (기본: gpt-5-mini)
#   --similarity_threshold N                         (기본: 1.0)
set -euo pipefail

TEACHER_DIR="${TEACHER_DIR:-data/teacher_GPT_5_Mini_user_GPT_5_Mini_0520_1133}"
SASM_STORE="${SASM_STORE:-sasm-memories/store.json}"
PREDICTOR_LLM="${PREDICTOR_LLM:-gpt-5-mini}"
SIMILARITY_THRESHOLD="${SIMILARITY_THRESHOLD:-1.0}"
PYBIN="${PYBIN:-python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --teacher_dir)           TEACHER_DIR="$2";           shift 2 ;;
    --sasm_store)            SASM_STORE="$2";             shift 2 ;;
    --predictor_llm)         PREDICTOR_LLM="$2";         shift 2 ;;
    --similarity_threshold)  SIMILARITY_THRESHOLD="$2";  shift 2 ;;
    *) echo "알 수 없는 옵션: $1"; exit 1 ;;
  esac
done

if [[ ! -d "$TEACHER_DIR/trajectories" ]]; then
  echo "ERROR: trajectories 디렉토리 없음: $TEACHER_DIR/trajectories"
  exit 1
fi

mkdir -p "$(dirname "$SASM_STORE")"

echo "=== SASM Memory Build (기존 trajectory 재사용) ==="
echo "TEACHER_DIR           : $TEACHER_DIR"
echo "SASM_STORE            : $SASM_STORE"
echo "PREDICTOR_LLM         : $PREDICTOR_LLM"
echo "SIMILARITY_THRESHOLD  : $SIMILARITY_THRESHOLD"
echo ""

cd "${TOOLSANDBOX_ROOT:?Set TOOLSANDBOX_ROOT to your ToolSandbox repo path}" && "$PYBIN" - <<PYEOF
import json
import sys
from pathlib import Path

teacher_dir = Path("$TEACHER_DIR")
sasm_store  = "$SASM_STORE"
predictor_llm = "$PREDICTOR_LLM"
similarity_threshold = float("$SIMILARITY_THRESHOLD")

# result_summary에서 similarity 로드
summary_path = teacher_dir / "result_summary.json"
sim_map = {}
if summary_path.exists():
    data = json.load(open(summary_path))
    for r in data.get("per_scenario_results", []):
        sim_map[r["name"]] = r.get("similarity", 0.0)

from tool_sandbox.common.execution_context import ExecutionContext
from tool_sandbox.sasm_memories.sasm_builder import SASMBuilder

builder = SASMBuilder(
    store_path=sasm_store,
    llm=predictor_llm,
    similarity_threshold=similarity_threshold,
    build_from_failures=False,
)

traj_root = teacher_dir / "trajectories"
scenarios = sorted(p.name for p in traj_root.iterdir() if p.is_dir())
print(f"총 {len(scenarios)}개 시나리오 처리 시작")

total_added = 0
for name in scenarios:
    ctx_path = traj_root / name / "execution_context.json"
    if not ctx_path.exists():
        print(f"[SKIP] {name}: execution_context.json 없음")
        continue
    ctx_data = json.load(open(ctx_path))
    context = ExecutionContext.from_dict(ctx_data)
    similarity = sim_map.get(name, 0.0)
    added = builder.build_from_scenario_result(
        scenario_name=name,
        context=context,
        similarity=similarity,
    )
    total_added += added

print(f"\n완료: 총 {total_added}개 항목 추가")
count = len(json.load(open(sasm_store))) if Path(sasm_store).exists() else 0
print(f"SASM store 총 항목 수: {count}  →  {sasm_store}")
PYEOF
