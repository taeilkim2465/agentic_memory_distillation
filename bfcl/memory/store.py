"""
Memory stores for BFCL — load, embed, and retrieve workflow/subtask/function memories.
Adapted from tau2-bench/src/tau2/memory/agent.py retrieval logic.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def _embed(text: str, model: str = "text-embedding-3-small") -> list[float]:
    import os
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.embeddings.create(model=model, input=[text])
    return [float(v) for v in resp.data[0].embedding]


def _cosine(a: list, b: list) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Workflow memory store
# ---------------------------------------------------------------------------

class WorkflowMemoryStore:
    """Load documents.json + embeddings.npy; retrieve top-k by cosine similarity."""

    def __init__(
        self,
        memory_dir: Path,
        embedding_model: str = "text-embedding-3-small",
    ):
        self._docs: list[dict] = []
        self._embeddings: Optional[np.ndarray] = None
        self._embedding_model = embedding_model

        docs_path = memory_dir / "workflow" / "documents.json"
        embs_path = memory_dir / "workflow" / "embeddings.npy"

        if not docs_path.exists():
            return

        with docs_path.open(encoding="utf-8") as f:
            self._docs = json.load(f)

        if embs_path.exists():
            try:
                arr = np.load(str(embs_path))
                if arr.ndim == 2 and arr.shape[0] == len(self._docs):
                    self._embeddings = arr
            except Exception:
                pass

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        category: str = "",
    ) -> list[dict]:
        if not self._docs:
            return []

        if category:
            pairs = [(i, d) for i, d in enumerate(self._docs) if d.get("category") == category]
        else:
            pairs = list(enumerate(self._docs))

        if not pairs:
            return []

        if self._embeddings is not None:
            query_vec = _embed(query, self._embedding_model)
            if query_vec is not None:
                q = np.array(query_vec, dtype=np.float32)
                q_norm = np.linalg.norm(q) + 1e-12
                indices = [i for i, _ in pairs]
                embs = self._embeddings[indices]
                d_norms = np.linalg.norm(embs, axis=1) + 1e-12
                sims = (embs @ q) / (d_norms * q_norm)
                order = np.argsort(sims)[::-1][:top_k]
                return [self._docs[indices[i]] for i in order]

        return [d for _, d in pairs[:top_k]]

    def format_for_prompt(self, docs: list[dict]) -> str:
        lines = []
        for i, d in enumerate(docs, 1):
            lines.append(f"{i}. [{d.get('task_id', '')}] {d.get('insight', '')}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Subtask segment store
# ---------------------------------------------------------------------------

class SubtaskSegmentStore:
    """Load segments.jsonl, pre-embed label+description, retrieve per subtask."""

    def __init__(
        self,
        memory_dir: Path,
        embedding_model: str = "text-embedding-3-small",
    ):
        self._segments: list[dict] = []
        self._embeddings: list[Optional[list[float]]] = []
        self._embedding_model = embedding_model

        seg_path = memory_dir / "subtask" / "segments.jsonl"
        emb_path = memory_dir / "subtask" / "segments_embeddings.npy"

        if not seg_path.exists():
            return

        for line in seg_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    self._segments.append(json.loads(line))
                except Exception:
                    pass

        if emb_path.exists():
            try:
                arr = np.load(str(emb_path))
                if arr.ndim == 2 and arr.shape[0] == len(self._segments):
                    self._embeddings = [arr[i].tolist() for i in range(len(self._segments))]
            except Exception:
                pass

        if not self._embeddings:
            for seg in self._segments:
                text = f"{seg.get('label', '')} {seg.get('description', '')}".strip()
                self._embeddings.append(_embed(text, embedding_model))

    def __len__(self) -> int:
        return len(self._segments)

    def retrieve_for_subtasks(
        self,
        subtasks: list[str],
        min_similarity: float = 0.45,
        category: str = "",
    ) -> list[dict]:
        if not self._segments:
            return []

        if category:
            pairs = [
                (seg, emb)
                for seg, emb in zip(self._segments, self._embeddings)
                if seg.get("category") == category
            ]
        else:
            pairs = list(zip(self._segments, self._embeddings))

        if not pairs:
            return []

        seen_labels: set[str] = set()
        results: list[dict] = []

        for subtask_text in subtasks:
            query_vec = _embed(subtask_text, self._embedding_model)
            if query_vec is None:
                continue

            best_score = -1.0
            best_seg = None
            for seg, seg_vec in pairs:
                if seg_vec is None:
                    continue
                score = _cosine(query_vec, seg_vec)
                if score > best_score:
                    best_score = score
                    best_seg = seg

            if best_seg is None or best_score < min_similarity:
                continue

            label = best_seg.get("label", str(id(best_seg)))
            if label in seen_labels:
                continue
            seen_labels.add(label)
            results.append(best_seg)

        return results

    @staticmethod
    def format_for_prompt(segments: list[dict]) -> str:
        lines = []
        for seg in segments:
            label = seg.get("label", "")
            desc = seg.get("description", "")
            tool_calls = seg.get("tool_calls", [])
            entry = f"- [{label}] {desc}"
            if tool_calls:
                entry += "\n  Tool Call Examples:"
                for call in tool_calls[:2]:
                    entry += f"\n    {call}"
            lines.append(entry)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Function memory store
# ---------------------------------------------------------------------------

class FunctionMemoryStore:
    """Load records.jsonl + embeddings.npy; retrieve by function name + query similarity."""

    def __init__(
        self,
        memory_dir: Path,
        embedding_model: str = "text-embedding-3-small",
    ):
        self._records: list[dict] = []
        self._embeddings: Optional[np.ndarray] = None
        self._embedding_model = embedding_model

        rec_path = memory_dir / "function" / "records.jsonl"
        emb_path = memory_dir / "function" / "records_embeddings.npy"

        if not rec_path.exists():
            return

        for line in rec_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    self._records.append(json.loads(line))
                except Exception:
                    pass

        if emb_path.exists():
            try:
                arr = np.load(str(emb_path))
                if arr.ndim == 2 and arr.shape[0] == len(self._records):
                    self._embeddings = arr
            except Exception:
                pass

    def retrieve(
        self,
        tool_name: str,
        turn_query: str = "",
        max_examples: int = 2,
    ) -> list[dict]:
        matched_indices = [
            i for i, r in enumerate(self._records)
            if r.get("tool_name") == tool_name
        ]
        if not matched_indices:
            return []

        if turn_query and self._embeddings is not None:
            try:
                query_vec = _embed(turn_query, self._embedding_model)
                if query_vec is not None:
                    q = np.array(query_vec, dtype=np.float32)
                    q_norm = np.linalg.norm(q) + 1e-12
                    embs = self._embeddings[matched_indices]
                    d_norms = np.linalg.norm(embs, axis=1) + 1e-12
                    sims = (embs @ q) / (d_norms * q_norm)
                    order = np.argsort(sims)[::-1]
                    matched_indices = [matched_indices[i] for i in order]
            except Exception:
                pass

        return [self._records[i] for i in matched_indices[:max_examples]]

    @staticmethod
    def format_hint(records: list[dict], tool_name: str) -> str:
        lines = []
        for r in records:
            think = r.get("think", "")
            if think:
                lines.append(f"[Why this function is called]\n{think}")
            args = r.get("arguments", {})
            result = str(r.get("result", ""))[:300]
            if args:
                arg_lines = "\n".join(f"    {k}: {v}" for k, v in args.items())
                call_str = f"{tool_name}(\n{arg_lines}\n)"
            else:
                call_str = f"{tool_name}()"
            lines.append(f"[Past successful call]\n{call_str}\n→ {result}")
        return "\n\n".join(lines)
