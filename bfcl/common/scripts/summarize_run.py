#!/usr/bin/env python3
"""
BFCL 실행 결과 요약 스크립트.

Usage:
    python run/summarize_run.py 0516_2020
    python run/summarize_run.py 0516_2020 --score-dir score/my_scores
    python run/summarize_run.py --all          # result/ 아래 모든 타임스탬프 나열
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO_ROOT / "result"
SCORE_ROOT = REPO_ROOT / "score"

TS_PATTERN = r"\d{4}_\d{4}"


def find_result_files(timestamp: str) -> list[tuple[str, Path]]:
    """Return (experiment_label, result_json_path) for all files matching the timestamp."""
    matches = []
    for result_json in RESULT_ROOT.rglob("*.json"):
        parts = result_json.parts
        # Find the timestamp segment in the path
        for i, part in enumerate(parts):
            if part == timestamp:
                # Path: result/{exp_label}/{model}/{timestamp}/...
                # parts[i-2] = exp_label, parts[i-1] = model
                exp_label = parts[i - 2] if i >= 2 else (parts[i - 1] if i >= 1 else "unknown")
                matches.append((exp_label, result_json))
                break
    return sorted(matches)


def find_score_file(model_name: str, test_category: str, score_dir: Path, exp_label: str = "") -> Path | None:
    """Look for a score file for the given model/category."""
    # Priority: explicit score_dir > score/{exp_label} > score/{model_name} (no cross-exp fallback)
    search_bases = []
    if score_dir != SCORE_ROOT:
        search_bases.append(score_dir)
    if exp_label:
        search_bases.append(SCORE_ROOT / exp_label)
    # Direct model path (no exp label) as last resort
    search_bases.append(SCORE_ROOT / model_name)

    for base in search_bases:
        direct = base / model_name / "multi_turn" / f"BFCL_v4_{test_category}_score.json"
        if direct.exists():
            return direct
        # Also handle base = score/{exp_label} → score/{exp_label}/{model}/multi_turn/...
        alt = base / "multi_turn" / f"BFCL_v4_{test_category}_score.json"
        if alt.exists():
            return alt
    return None


def summarize_result_file(exp_label: str, result_path: Path, score_dir: Path) -> dict:
    lines = result_path.read_text(encoding="utf-8").splitlines()
    lines = [l for l in lines if l.strip()]

    total = len(lines)
    ids = []
    for line in lines:
        try:
            obj = json.loads(line)
            if "id" in obj:
                ids.append(obj["id"])
        except Exception:
            pass

    # Infer names from path
    # structure: result/{exp_label}/{timestamp}/{model_name}/multi_turn/{file}.json
    parts = result_path.parts
    model_name = "unknown"
    test_category = "unknown"
    for i, p in enumerate(parts):
        if p == "multi_turn" and i > 0:
            model_name = parts[i - 1]
        if p.endswith("_result.json"):
            test_category = p.replace("BFCL_v4_", "").replace("_result.json", "")

    # Try to read test_category from filename
    fname = result_path.stem  # e.g. BFCL_v4_multi_turn_base_result
    if fname.startswith("BFCL_v4_") and fname.endswith("_result"):
        test_category = fname[len("BFCL_v4_"):-len("_result")]

    info = {
        "experiment": exp_label,
        "model": model_name,
        "test_category": test_category,
        "completed": total,
        "result_path": str(result_path.relative_to(REPO_ROOT)),
    }

    # Try to find matching score file
    score_path = find_score_file(model_name, test_category, score_dir, exp_label=exp_label)
    if score_path and score_path.exists():
        score_lines = [l for l in score_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if score_lines:
            try:
                summary = json.loads(score_lines[0])
                if "accuracy" in summary:
                    info["accuracy"] = summary["accuracy"]
                    info["correct"] = summary.get("correct_count", "?")
                    info["total_scored"] = summary.get("total_count", "?")
                    info["score_path"] = str(score_path.relative_to(REPO_ROOT))
            except Exception:
                pass

    return info


def print_summary(info: dict) -> None:
    print(f"\n{'─' * 52}")
    print(f"  Experiment  : {info['experiment']}")
    print(f"  Model       : {info['model']}")
    print(f"  Category    : {info['test_category']}")
    print(f"  Completed   : {info['completed']} tasks")
    print(f"  Result file : {info['result_path']}")
    if "accuracy" in info:
        acc_pct = f"{info['accuracy'] * 100:.1f}%"
        print(f"  Accuracy    : {acc_pct}  ({info['correct']}/{info['total_scored']})")
        print(f"  Score file  : {info.get('score_path', '')}")
    else:
        print(f"  Accuracy    : (평가 미실행 — bfcl_eval evaluate 실행 필요)")
    print(f"{'─' * 52}")


def list_all_timestamps() -> None:
    import re
    pattern = re.compile(r"\d{4}_\d{4}")
    # {timestamp: [(exp_label, result_path), ...]}
    grouped: dict[str, list[tuple[str, Path]]] = {}
    for p in RESULT_ROOT.rglob("*.json"):
        for i, part in enumerate(p.parts):
            if pattern.fullmatch(part):
                exp_label = p.parts[i - 2] if i >= 2 else (p.parts[i - 1] if i >= 1 else "unknown")
                grouped.setdefault(part, []).append((exp_label, p))
                break

    if not grouped:
        print("  (결과 없음)")
        return

    for ts in sorted(grouped):
        entries = grouped[ts]
        for exp_label, result_path in entries:
            n = len([l for l in result_path.read_text().splitlines() if l.strip()])
            print(f"  {ts}  [{exp_label}]  {n}개 완료  —  {result_path.relative_to(REPO_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="BFCL 실행 결과 요약")
    parser.add_argument("timestamp", nargs="?", help="타임스탬프 (예: 0516_2020)")
    parser.add_argument("--score-dir", default=None, help="스코어 디렉토리 (기본: score/)")
    parser.add_argument("--all", action="store_true", help="모든 타임스탬프 목록 출력")
    args = parser.parse_args()

    score_dir = Path(args.score_dir) if args.score_dir else SCORE_ROOT
    if not score_dir.is_absolute():
        score_dir = REPO_ROOT / score_dir

    if args.all:
        print("\n사용 가능한 타임스탬프:")
        list_all_timestamps()
        return

    if not args.timestamp:
        parser.print_help()
        sys.exit(1)

    matches = find_result_files(args.timestamp)
    if not matches:
        print(f"[ERROR] '{args.timestamp}' 에 해당하는 결과 파일을 찾을 수 없습니다.")
        print("사용 가능한 타임스탬프 목록: python run/summarize_run.py --all")
        sys.exit(1)

    for exp_label, result_path in matches:
        info = summarize_result_file(exp_label, result_path, score_dir)
        print_summary(info)

    if len(matches) > 1:
        print(f"\n총 {len(matches)}개 결과 파일")


if __name__ == "__main__":
    main()
