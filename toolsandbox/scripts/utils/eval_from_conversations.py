"""Compute evaluation metrics from existing conversation.json files (no result_summary.json needed)."""
import json
import glob
import os
import sys
from collections import defaultdict


def compute_similarity_from_conversation(conv_path: str) -> dict:
    data = json.load(open(conv_path))

    # Collect best milestone_similarity per milestone_index across all turns
    milestone_best: dict[int, float] = {}
    minefield_best: dict[int, float] = {}

    for turn in data:
        for key in turn:
            if not isinstance(turn[key], dict):
                continue
            details = turn[key]

            for match in details.get("milestone_matches", []):
                idx = match["milestone_index"]
                sim = match["milestone_similarity"]
                milestone_best[idx] = max(milestone_best.get(idx, 0.0), sim)

            for match in details.get("minefield_matches", []):
                idx = match["minefield_index"]
                sim = match["minefield_similarity"]
                minefield_best[idx] = max(minefield_best.get(idx, 0.0), sim)

    milestone_similarity = (
        sum(milestone_best.values()) / len(milestone_best) if milestone_best else 1.0
    )
    minefield_similarity = (
        sum(minefield_best.values()) / len(minefield_best) if minefield_best else 0.0
    )
    similarity = (int(minefield_similarity == 0)) * milestone_similarity

    return {
        "milestone_similarity": milestone_similarity,
        "minefield_similarity": minefield_similarity,
        "similarity": similarity,
    }


def summarize_run(run_dir: str, max_scenarios: int = 200) -> None:
    convs = sorted(
        glob.glob(f"{run_dir}/trajectories/*/conversation.json")
    )[:max_scenarios]

    if not convs:
        print(f"{run_dir}: conversation.json 없음")
        return

    results = []
    for f in convs:
        scenario = f.split("/")[-2]
        try:
            r = compute_similarity_from_conversation(f)
            r["name"] = scenario
            results.append(r)
        except Exception as e:
            results.append({"name": scenario, "similarity": 0.0,
                            "milestone_similarity": 0.0, "minefield_similarity": 0.0})

    n = len(results)
    avg_sim = sum(r["similarity"] for r in results) / n
    avg_ms  = sum(r["milestone_similarity"] for r in results) / n
    avg_mf  = sum(r["minefield_similarity"] for r in results) / n

    name = os.path.basename(run_dir)
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"  scenarios : {n}")
    print(f"  similarity          : {avg_sim:.4f}")
    print(f"  milestone_similarity: {avg_ms:.4f}")
    print(f"  minefield_similarity: {avg_mf:.4f}")


if __name__ == "__main__":
    dirs = sys.argv[1:] if len(sys.argv) > 1 else []
    if not dirs:
        print("Usage: python eval_from_conversations.py <run_dir> [run_dir ...]")
        sys.exit(1)
    for d in sorted(dirs):
        summarize_run(d, max_scenarios=200)
