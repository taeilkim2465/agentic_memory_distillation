#!/usr/bin/env python3
"""
Build BFCL teacher memory from one or more teacher result files.

Usage:
    python run/build_memory_from_results.py \\
        --result-files result/0514_1230/claude-sonnet-4-6/multi_turn_base_result.json \\
        --memory-dir data/memory/0514 \\
        --teacher-llm claude-sonnet-4-6 \\
        --test-category multi_turn_base

The script:
  1. Loads each result file (BFCL result JSON format).
  2. Loads the corresponding test entries to get question/function data.
  3. Calls BFCLMemoryBuilder to extract workflow, subtask, and function memories.
  4. Saves to memory_dir/{workflow,subtask,function}/.
"""
import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

# Ensure bfcl_eval is importable from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bfcl_eval.constants.eval_config import PROJECT_ROOT
from bfcl_eval.memory.builder import BFCLMemoryBuilder
from bfcl_eval.utils import load_dataset_entry, parse_test_category_argument


def main():
    parser = argparse.ArgumentParser(description="Build BFCL teacher memory from result files.")
    parser.add_argument(
        "--result-files",
        nargs="+",
        required=True,
        help="One or more BFCL result JSON files (teacher outputs).",
    )
    parser.add_argument(
        "--memory-dir",
        required=True,
        help="Output directory for memory files (will be created if missing).",
    )
    parser.add_argument(
        "--teacher-llm",
        default="gpt-5-mini-2025-08-07",
        help="LLM used to extract workflow and subtask memories (default: gpt-5-mini-2025-08-07).",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="Embedding model for memory retrieval (default: text-embedding-3-small).",
    )
    parser.add_argument(
        "--test-category",
        nargs="+",
        default=["multi_turn_base"],
        help="Test categories whose entries are needed (default: multi_turn_base).",
    )
    parser.add_argument(
        "--include-force-quit",
        action="store_true",
        default=False,
        help="Also process entries with force_quit generation status (default: skip).",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["workflow", "subtask", "function"],
        default=None,
        help="Build only the specified memory type(s). Default: all.",
    )
    parser.add_argument(
        "--score-file",
        default=None,
        help=(
            "BFCL score JSON file. If provided, only tasks that succeeded "
            "(i.e. whose IDs appear in the result file but NOT in the score file) "
            "are included in memory."
        ),
    )
    parser.add_argument(
        "--teacher-llm-base-url",
        "--vllm-url",
        default=None,
        help="Base URL for local OpenAI-compatible server (e.g. http://localhost:8001/v1). "
             "When set, use 'openai/<model>' as --teacher-llm.",
    )
    parser.add_argument(
        "--teacher-llm-api-key",
        default="not-needed",
        help="API key for local server (default: 'not-needed').",
    )
    args = parser.parse_args()

    memory_dir = Path(args.memory_dir)
    print(f"Memory output dir : {memory_dir}")
    print(f"Teacher LLM       : {args.teacher_llm}")
    print(f"Embedding model   : {args.embedding_model}")

    # Load all test entries (needed to get question / function docs)
    print("\nLoading test entries...")
    test_entries_by_id: dict[str, dict] = {}
    for cat in parse_test_category_argument(args.test_category):
        entries = load_dataset_entry(cat)
        for e in entries:
            test_entries_by_id[e["id"]] = e
    print(f"  {len(test_entries_by_id)} entries loaded.")

    builder = BFCLMemoryBuilder(
        memory_dir=memory_dir,
        teacher_llm=args.teacher_llm,
        embedding_model=args.embedding_model,
        teacher_llm_base_url=args.teacher_llm_base_url,
        teacher_llm_api_key=args.teacher_llm_api_key,
    )

    build_types = set(args.only) if args.only else None
    if build_types:
        print(f"Building only    : {', '.join(sorted(build_types))}")

    # Derive successful task IDs from score file if provided.
    # BFCL score files record only failed tasks; success = result_ids - failed_ids.
    allowed_ids: set[str] | None = None
    if args.score_file:
        score_path = Path(args.score_file)
        if not score_path.exists():
            print(f"[ERROR] Score file not found: {score_path}")
            sys.exit(1)
        failed_ids: set[str] = set()
        with score_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "id" in obj:
                    failed_ids.add(obj["id"])
        # Collect all result IDs across provided result files
        all_result_ids: set[str] = set()
        for rf in args.result_files:
            rf_path = Path(rf)
            if rf_path.exists():
                with rf_path.open(encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            obj = json.loads(line)
                            if "id" in obj:
                                all_result_ids.add(obj["id"])
        allowed_ids = all_result_ids - failed_ids
        print(f"Score file       : {score_path}")
        print(f"  Total result   : {len(all_result_ids)}")
        print(f"  Failed (skip)  : {len(failed_ids)}")
        print(f"  Success (use)  : {len(allowed_ids)}")

    total = {"workflow": 0, "subtask": 0, "function": 0}
    for rf in args.result_files:
        rf = Path(rf)
        if not rf.exists():
            print(f"[WARN] Result file not found: {rf} — skipping.")
            continue
        print(f"\nProcessing: {rf}")
        counts = builder.build_from_result_file(
            result_file=rf,
            test_entries_by_id=test_entries_by_id,
            only_ok_status=not args.include_force_quit,
            build_types=build_types,
            allowed_ids=allowed_ids,
        )
        for k, v in counts.items():
            total[k] += v
        print(f"  → workflow={counts['workflow']}  subtask={counts['subtask']}  function={counts['function']}")

    print(f"\n{'='*50}")
    print(f" Total memories built:")
    print(f"  Workflow  : {total['workflow']}")
    print(f"  Subtask   : {total['subtask']} segments")
    print(f"  Function  : {total['function']} records")
    print(f" Memory dir : {memory_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
