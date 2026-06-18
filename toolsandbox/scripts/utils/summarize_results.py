#!/usr/bin/env python3
"""Aggregate scores from ToolSandbox run directories.

Usage:
    # 특정 run 하나
    python scripts/summarize_results.py data/teacher_GPT_5_Mini_user_GPT_5_Mini_05_17_2026_16_06_46

    # 여러 run 비교
    python scripts/summarize_results.py data/teacher_* data/student_* data/agent_*

    # data/ 아래 전부
    python scripts/summarize_results.py data/*

    # 시간 태그로 필터 (예: 0519_1945 태그가 포함된 run만)
    python scripts/summarize_results.py data/* --tag 0519_1945
    python scripts/summarize_results.py data/student_* --tag 0519
"""

import argparse
import json
from pathlib import Path
from typing import Optional


def load_scenario_result(traj_dir: Path) -> Optional[dict]:
    """conversation.json에서 시나리오 결과를 추출한다."""
    conv_path = traj_dir / "conversation.json"
    if not conv_path.exists():
        return None

    data = json.loads(conv_path.read_text(encoding="utf-8"))

    # milestone/minefield 매칭 수집
    milestone_sims: list[float] = []
    minefield_sims: list[float] = []

    for item in data:
        ad = item.get("assistant_details", {})
        for m in ad.get("milestone_matches", []):
            s = m.get("milestone_similarity")
            if s is not None:
                milestone_sims.append(s)
        for m in ad.get("minefield_matches", []):
            s = m.get("milestone_similarity")
            if s is not None:
                minefield_sims.append(s)

    milestone_similarity = (
        sum(milestone_sims) / len(milestone_sims) if milestone_sims else 1.0
    )
    minefield_similarity = (
        sum(minefield_sims) / len(minefield_sims) if minefield_sims else 0.0
    )
    # minefield가 하나라도 매칭되면 0으로 강제
    similarity = 0.0 if minefield_similarity > 0 else milestone_similarity

    # 대화 턴 수 (system 제외)
    turn_count = sum(
        1 for item in data if item.get("role") in ("user", "assistant")
    )

    return {
        "scenario": traj_dir.name,
        "milestone_similarity": milestone_similarity,
        "minefield_similarity": minefield_similarity,
        "similarity": similarity,
        "turn_count": turn_count,
        "success": similarity == 1.0,
    }


def summarize_run(run_dir: Path) -> dict:
    """run 디렉토리 하나를 집계한다.

    result_summary.json이 있으면 그것을 우선 사용한다 (정확).
    없으면 conversation.json에서 재구성한다 (근사치).
    """
    # result_summary.json 우선 사용
    summary_path = run_dir / "result_summary.json"
    if summary_path.exists():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        results = [
            {
                "scenario": r["name"],
                "milestone_similarity": r["milestone_similarity"],
                "minefield_similarity": r["minefield_similarity"],
                "similarity": r["similarity"],
                "turn_count": r["turn_count"],
                "success": r["similarity"] == 1.0,
            }
            for r in data["per_scenario_results"]
        ]
    else:
        traj_root = run_dir / "trajectories"
        if not traj_root.exists():
            traj_root = run_dir
        results = []
        for traj_dir in sorted(traj_root.iterdir()):
            if not traj_dir.is_dir():
                continue
            r = load_scenario_result(traj_dir)
            if r:
                results.append(r)

    if not results:
        return {"run": run_dir.name, "n": 0}

    n = len(results)
    avg_sim = sum(r["similarity"] for r in results) / n
    avg_mile = sum(r["milestone_similarity"] for r in results) / n
    avg_mine = sum(r["minefield_similarity"] for r in results) / n
    n_success = sum(1 for r in results if r["success"])
    avg_turns = sum(r["turn_count"] for r in results) / n

    return {
        "run": run_dir.name,
        "n": n,
        "similarity": avg_sim,
        "milestone_similarity": avg_mile,
        "minefield_similarity": avg_mine,
        "success_rate": n_success / n,
        "n_success": n_success,
        "avg_turns": avg_turns,
        "per_scenario": results,
    }


def print_summary(summary: dict, verbose: bool = False) -> None:
    print(f"\n{'='*70}")
    print(f"Run : {summary['run']}")
    print(f"{'='*70}")
    n = summary.get("n", 0)
    if n == 0:
        print("  결과 없음")
        return

    print(f"  시나리오 수     : {n}")
    print(f"  similarity      : {summary['similarity']:.4f}")
    print(f"    milestone     : {summary['milestone_similarity']:.4f}")
    print(f"    minefield     : {summary['minefield_similarity']:.4f}")
    print(f"  완전 성공 (1.0) : {summary['n_success']} / {n}  ({summary['success_rate']*100:.1f}%)")
    print(f"  평균 턴 수      : {summary['avg_turns']:.1f}")

    if verbose:
        print(f"\n  {'시나리오':<60} {'similarity':>10} {'success':>8}")
        print(f"  {'-'*60} {'-'*10} {'-'*8}")
        for r in summary["per_scenario"]:
            mark = "✓" if r["success"] else " "
            print(f"  {r['scenario'][:60]:<60} {r['similarity']:>10.4f} {mark:>8}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="*", type=Path, help="Run 디렉토리 경로 (생략 시 ./data/* 자동 사용)")
    parser.add_argument("-v", "--verbose", action="store_true", help="시나리오별 점수 출력")
    parser.add_argument("--tag", type=str, default=None, help="타임스탬프 태그 필터 (예: 0519_1945)")
    args = parser.parse_args()

    run_dirs = args.run_dirs or sorted(Path("data").iterdir())

    summaries = []
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue
        if args.tag and args.tag not in run_dir.name:
            continue
        s = summarize_run(run_dir)
        summaries.append(s)
        print_summary(s, verbose=args.verbose)

    if len(summaries) > 1:
        print(f"\n{'='*70}")
        print("비교 요약")
        print(f"{'='*70}")
        print(f"  {'Run':<50} {'similarity':>10} {'success%':>9} {'n':>5}")
        print(f"  {'-'*50} {'-'*10} {'-'*9} {'-'*5}")
        for s in summaries:
            if s.get("n", 0) == 0:
                continue
            print(
                f"  {s['run'][:50]:<50} "
                f"{s['similarity']:>10.4f} "
                f"{s['success_rate']*100:>8.1f}% "
                f"{s['n']:>5}"
            )


if __name__ == "__main__":
    main()
