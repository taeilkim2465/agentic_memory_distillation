"""Rebuild memory from saved trajectory execution_context.json files.

Reads result_summary.json to find successful scenarios (similarity == 1.0),
then reconstructs ExecutionContext from each trajectory's execution_context.json
and re-runs the memory builder.

Usage:
    python scripts/rebuild_memory_from_trajectories.py <run_dir> <memory_dir> [--threshold 1.0] [--builder-llm gpt-5-mini] [--clear]

Example:
    python scripts/rebuild_memory_from_trajectories.py \\
        data/teacher_GPT_5_Mini_user_GPT_5_Mini_0520_1133 \\
        memories/teacher_GPT_5_Mini_user_GPT_5_Mini_0520_1133 \\
        --clear
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tool_sandbox.common.execution_context import ExecutionContext
from tool_sandbox.memory.builder import ToolSandboxMemoryBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Teacher run 디렉토리")
    parser.add_argument("memory_dir", type=Path, help="Memory 저장 디렉토리")
    parser.add_argument("--threshold", type=float, default=1.0, help="similarity threshold (default: 1.0)")
    parser.add_argument("--builder-llm", default="gpt-5-mini", help="Memory builder LLM (default: gpt-5-mini)")
    parser.add_argument("--clear", action="store_true", help="기존 memory 삭제 후 재빌드")
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    memory_dir: Path = args.memory_dir

    if not run_dir.exists():
        print(f"run_dir 없음: {run_dir}")
        sys.exit(1)

    summary_path = run_dir / "result_summary.json"
    if not summary_path.exists():
        print(f"result_summary.json 없음: {summary_path}")
        sys.exit(1)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    all_results = summary["per_scenario_results"]
    successes = [r for r in all_results if r["similarity"] >= args.threshold]
    print(f"전체: {len(all_results)}개, similarity>={args.threshold}: {len(successes)}개")

    if args.clear and memory_dir.exists():
        shutil.rmtree(memory_dir)
        print(f"기존 memory 삭제: {memory_dir}")

    memory_dir.mkdir(parents=True, exist_ok=True)

    builder = ToolSandboxMemoryBuilder(
        memory_dir=memory_dir,
        builder_llm=args.builder_llm,
        think_model=args.builder_llm,
    )

    total_counts: dict[str, int] = {"workflow": 0, "subtask": 0, "function": 0}
    failed = []

    for i, result in enumerate(successes, 1):
        name = result["name"]
        categories = result.get("categories", [])
        traj_dir = run_dir / "trajectories" / name
        ctx_path = traj_dir / "execution_context.json"

        if not ctx_path.exists():
            print(f"[{i}/{len(successes)}] ✗ {name} → execution_context.json 없음")
            failed.append(name)
            continue

        try:
            ctx_dict = json.loads(ctx_path.read_text(encoding="utf-8"))
            context = ExecutionContext.from_dict(ctx_dict)
            counts = builder.build_from_scenario_result(
                scenario_name=name,
                categories=categories,
                context=context,
            )
            for k in total_counts:
                total_counts[k] += counts.get(k, 0)
            print(f"[{i}/{len(successes)}] ✓ {name} → {counts}")
        except Exception as e:
            print(f"[{i}/{len(successes)}] ✗ {name} → ERROR: {type(e).__name__}: {e}")
            failed.append(name)

    print(f"\n=== 완료 ===")
    print(f"  workflow : {total_counts['workflow']}개")
    print(f"  subtask  : {total_counts['subtask']}개")
    print(f"  function : {total_counts['function']}개")
    if failed:
        print(f"  실패     : {len(failed)}개 → {failed}")


if __name__ == "__main__":
    main()
