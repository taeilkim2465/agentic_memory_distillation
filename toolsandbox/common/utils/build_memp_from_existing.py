#!/usr/bin/env python3
"""기존 teacher 실행 결과(execution_context.json)에서 MEMP store를 빌드한다.

사용:
    python scripts/build_memp_from_existing.py \
        --run_dir data/agent_GPT_5_Mini_user_GPT_5_Mini_MMDD_HHMM \
        --memp_store memp-memories/store.json \
        [--keyword_llm gpt-5-mini] \
        [--proc_llm gpt-5-mini] \
        [--adjust_llm gpt-5-mini] \
        [--parallel 8]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tool_sandbox.common.execution_context import ExecutionContext
from tool_sandbox.memp_memories.memp_builder import MempBuilder


def _process_one(
    args: tuple[str, Path, float],
    *,
    memp_store: Path,
    keyword_llm: str,
    proc_llm: str,
    adjust_llm: str,
    similarity_threshold: float,
) -> tuple[str, int]:
    name, ctx_path, similarity = args
    try:
        with open(ctx_path) as f:
            ctx_data = json.load(f)
        context = ExecutionContext.from_dict(ctx_data)
    except Exception as e:
        print(f"[SKIP] {name}: context 로드 실패 — {e}", flush=True)
        return name, 0

    builder = MempBuilder(
        store_path=memp_store,
        keyword_llm=keyword_llm,
        proceduralize_llm=proc_llm,
        adjust_llm=adjust_llm,
        similarity_threshold=similarity_threshold,
        build_from_failures=True,
    )
    n = builder.build_from_scenario_result(
        scenario_name=name,
        context=context,
        similarity=similarity,
    )
    outcome = "success" if similarity >= similarity_threshold else "failure"
    print(f"[{'✓' if outcome == 'success' else '✗'}] {name} (sim={similarity:.2f}) → +{n} entry", flush=True)
    return name, n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True, help="teacher 실행 디렉토리")
    parser.add_argument("--memp_store", type=Path, default=Path("memp-memories/store.json"))
    parser.add_argument("--keyword_llm", type=str, default="gpt-5-mini")
    parser.add_argument("--proc_llm", type=str, default="gpt-5-mini")
    parser.add_argument("--adjust_llm", type=str, default="gpt-5-mini")
    parser.add_argument("--similarity_threshold", type=float, default=1.0)
    parser.add_argument("--parallel", "-p", type=int, default=8)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    result_summary_path = run_dir / "result_summary.json"

    if not result_summary_path.exists():
        print(f"result_summary.json 없음: {result_summary_path}", file=sys.stderr)
        sys.exit(1)

    with open(result_summary_path) as f:
        summary = json.load(f)

    name_to_sim: dict[str, float] = {
        r["name"]: r["similarity"] for r in summary["per_scenario_results"]
    }

    traj_root = run_dir / "trajectories"
    tasks: list[tuple[str, Path, float]] = []
    for name, sim in name_to_sim.items():
        ctx_path = traj_root / name / "execution_context.json"
        if not ctx_path.exists():
            print(f"[SKIP] {name}: execution_context.json 없음", flush=True)
            continue
        tasks.append((name, ctx_path, sim))

    print("=== build_memp_from_existing ===")
    print(f"run_dir   : {run_dir}")
    print(f"memp_store: {args.memp_store}")
    print(f"총 {len(tasks)}개 시나리오 처리 예정")
    print()

    args.memp_store.parent.mkdir(parents=True, exist_ok=True)
    fn = partial(
        _process_one,
        memp_store=args.memp_store,
        keyword_llm=args.keyword_llm,
        proc_llm=args.proc_llm,
        adjust_llm=args.adjust_llm,
        similarity_threshold=args.similarity_threshold,
    )

    if args.parallel > 1 and len(tasks) > 1:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(min(args.parallel, len(tasks))) as pool:
            results = pool.map(fn, tasks)
    else:
        results = [fn(t) for t in tasks]

    total = sum(n for _, n in results)
    succeeded = sum(1 for _, n in results if n > 0)
    print()
    print("=== 완료 ===")
    print(f"entries 생성: {succeeded}/{len(tasks)}개 시나리오")
    print(f"store 경로  : {args.memp_store}")


if __name__ == "__main__":
    main()
