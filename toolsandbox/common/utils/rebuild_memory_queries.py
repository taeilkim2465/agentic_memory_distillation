"""Rebuild memory query fields and embeddings from scenario definitions.

The original memory was built with task_query = first USER→AGENT message,
which is always the same generic string ("I want to send a message to someone.").
This script replaces each entry's query with the actual task instruction from the
SYSTEM→USER message in the scenario definition, then recomputes embeddings.

Usage:
    python scripts/rebuild_memory_queries.py <memory_dir>

Example:
    python scripts/rebuild_memory_queries.py memories/teacher_GPT_5_Mini_user_GPT_5_Mini_0519_1927
"""

import json
import sys
from pathlib import Path

import numpy as np
from openai import OpenAI

from tool_sandbox.common.execution_context import DatabaseNamespace, RoleType
from tool_sandbox.common.tool_discovery import ToolBackend
from tool_sandbox.memory.trajectory import extract_task_instruction
from tool_sandbox.scenarios import named_scenarios

EMBED_MODEL = "text-embedding-3-small"


def build_scenario_query_map() -> dict[str, str]:
    """Return {scenario_name: task_instruction} for all known scenarios."""
    print("시나리오 정의 로딩 중...")
    scenarios = named_scenarios(preferred_tool_backend=ToolBackend.DEFAULT)
    print(f"  총 {len(scenarios)}개 시나리오 로드됨")

    query_map: dict[str, str] = {}
    for name, scenario in scenarios.items():
        try:
            db = scenario.starting_context.get_database(
                DatabaseNamespace.SANDBOX,
                get_all_history_snapshots=True,
                drop_sandbox_message_index=False,
                drop_headguard=True,
            )
            rows = db.to_dicts()
            for row in rows:
                if row.get("sender") == RoleType.SYSTEM and row.get("recipient") == RoleType.USER:
                    task = extract_task_instruction(row["content"])
                    if task:
                        query_map[name] = task
                    # Don't break: keep overwriting to get the LAST SYSTEM→USER
                    # (earlier ones are few-shot examples)
        except Exception as e:
            pass  # skip scenarios that fail to load
    print(f"  task instruction 추출됨: {len(query_map)}개")
    return query_map


def embed_texts(texts: list[str], client: OpenAI) -> np.ndarray:
    """Embed a list of texts in batches, return (N, D) float32 array."""
    all_embs = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        batch_embs = [d.embedding for d in resp.data]
        all_embs.extend(batch_embs)
        print(f"  임베딩: {min(i + batch_size, len(texts))}/{len(texts)}")
    return np.array(all_embs, dtype=np.float32)


def rebuild_workflow(memory_dir: Path, query_map: dict[str, str], client: OpenAI) -> None:
    docs_path = memory_dir / "workflow" / "documents.json"
    emb_path = memory_dir / "workflow" / "embeddings.npy"
    if not docs_path.exists():
        print("[workflow] documents.json 없음, 스킵")
        return

    docs = json.loads(docs_path.read_text(encoding="utf-8"))
    updated = 0
    not_found = []
    for doc in docs:
        name = doc.get("scenario_name", "")
        if name in query_map:
            doc["query"] = query_map[name]
            updated += 1
        else:
            not_found.append(name)

    print(f"[workflow] {updated}/{len(docs)}개 query 업데이트, 미발견: {len(not_found)}개")
    if not_found:
        print(f"  미발견 시나리오 예시: {not_found[:3]}")

    print("[workflow] embedding 재계산 중...")
    queries = [doc["query"] for doc in docs]
    embs = embed_texts(queries, client)

    docs_path.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(str(emb_path), embs)
    print(f"[workflow] 저장 완료: {docs_path}")


def rebuild_subtask(memory_dir: Path, query_map: dict[str, str], client: OpenAI) -> None:
    segs_path = memory_dir / "subtask" / "segments.jsonl"
    emb_path = memory_dir / "subtask" / "embeddings.npy"
    if not segs_path.exists():
        print("[subtask] segments.jsonl 없음, 스킵")
        return

    segments = [
        json.loads(line)
        for line in segs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    updated = 0
    for seg in segments:
        name = seg.get("scenario_name", "")
        if name in query_map:
            seg["query"] = query_map[name]
            updated += 1

    print(f"[subtask] {updated}/{len(segments)}개 query 업데이트")
    print("[subtask] embedding 재계산 중 (label + description 기준)...")
    texts = [seg.get("label", "") + ": " + seg.get("description", "") for seg in segments]
    embs = embed_texts(texts, client)

    segs_path.write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in segments) + "\n",
        encoding="utf-8",
    )
    np.save(str(emb_path), embs)
    print(f"[subtask] 저장 완료: {segs_path}")


def rebuild_function(memory_dir: Path, query_map: dict[str, str]) -> None:
    records_path = memory_dir / "function" / "records.jsonl"
    if not records_path.exists():
        print("[function] records.jsonl 없음, 스킵")
        return

    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    updated = 0
    for rec in records:
        name = rec.get("scenario_name", "")
        if name in query_map:
            rec["task_query"] = query_map[name]
            updated += 1

    print(f"[function] {updated}/{len(records)}개 task_query 업데이트")
    records_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    print(f"[function] 저장 완료 (function은 local embed 사용, 별도 재계산 불필요)")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python rebuild_memory_queries.py <memory_dir>")
        sys.exit(1)

    memory_dir = Path(sys.argv[1])
    if not memory_dir.exists():
        print(f"경로 없음: {memory_dir}")
        sys.exit(1)

    print(f"메모리 디렉토리: {memory_dir}\n")

    query_map = build_scenario_query_map()

    client = OpenAI()

    print("\n=== workflow 메모리 재빌드 ===")
    rebuild_workflow(memory_dir, query_map, client)

    print("\n=== subtask 메모리 재빌드 ===")
    rebuild_subtask(memory_dir, query_map, client)

    print("\n=== function 메모리 재빌드 ===")
    rebuild_function(memory_dir, query_map)

    print("\n완료!")


if __name__ == "__main__":
    main()
