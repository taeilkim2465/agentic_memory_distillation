# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Load and retrieve memory entries from disk."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from openai import OpenAI

from tool_sandbox.memory.data_model import (
    FunctionStepRecord,
    SubtaskSegment,
    WorkflowMemory,
)

LOGGER = logging.getLogger(__name__)

_openai_client: Optional[OpenAI] = None
_DEFAULT_EMBED_MODEL = "text-embedding-3-small"


def _get_client() -> OpenAI:
    """Return an OpenAI client that always targets the real OpenAI API for embeddings."""
    global _openai_client
    if _openai_client is None:
        import os
        # Explicitly set base_url so OPENAI_BASE_URL (pointing to vllm) is not used
        embed_base_url = os.environ.get("OPENAI_EMBED_BASE_URL", "https://api.openai.com/v1")
        _openai_client = OpenAI(base_url=embed_base_url)
    return _openai_client


def _embed_query(text: str, stored_dim: Optional[int] = None) -> np.ndarray:
    """Embed a query string, using the same dimensionality as stored embeddings."""
    if stored_dim == 256:
        return _local_embed(text)
    try:
        resp = _get_client().embeddings.create(model=_DEFAULT_EMBED_MODEL, input=text)
        return np.array(resp.data[0].embedding, dtype=np.float32)
    except Exception:
        return _local_embed(text)



def _local_embed(text: str, dim: int = 256) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    text = text.lower()
    for n in (2, 3):
        for i in range(len(text) - n + 1):
            vec[hash(text[i : i + n]) % dim] += 1.0
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def _cosine_sim(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms > 0, norms, 1.0)
    normed = matrix / safe_norms
    q_norm = query / (float(np.linalg.norm(query)) or 1.0)
    return normed @ q_norm


class WorkflowMemoryStore:
    """Stores workflow-level memories; retrieved by semantic similarity to the task query."""

    def __init__(self, memory_dir: Path) -> None:
        self.documents: list[WorkflowMemory] = []
        self.embeddings: Optional[np.ndarray] = None
        self._dim: Optional[int] = None

        docs_path = memory_dir / "workflow" / "documents.json"
        emb_path = memory_dir / "workflow" / "embeddings.npy"

        if docs_path.exists():
            raw = json.loads(docs_path.read_text(encoding="utf-8"))
            self.documents = [WorkflowMemory.from_dict(d) for d in raw]
        if emb_path.exists() and self.documents:
            self.embeddings = np.load(str(emb_path))
            if self.embeddings.ndim == 2 and len(self.embeddings) > 0:
                self._dim = self.embeddings.shape[1]

    def retrieve(self, query: str, top_k: int = 1, min_sim: float = 0.3) -> list[WorkflowMemory]:
        if not self.documents or self.embeddings is None:
            return []
        q = _embed_query(query, self._dim)
        sims = _cosine_sim(q, self.embeddings)
        indices = np.argsort(sims)[::-1][:top_k]
        return [self.documents[i] for i in indices if sims[i] >= min_sim]

    @staticmethod
    def format_for_prompt(memories: list[WorkflowMemory]) -> str:
        if not memories:
            return ""
        lines = ["## Workflow Hint (advisory — adapt to the actual task)"]
        for m in memories:
            if m.involved_tools:
                lines.append(f"Tools involved: {', '.join(m.involved_tools)}")
            lines.append(m.insight)
        return "\n".join(lines)


class SubtaskSegmentStore:
    """Stores subtask segments; retrieved by semantic similarity to decomposed subtask labels."""

    def __init__(self, memory_dir: Path) -> None:
        self.segments: list[SubtaskSegment] = []
        self.embeddings: Optional[np.ndarray] = None
        self._dim: Optional[int] = None

        segs_path = memory_dir / "subtask" / "segments.jsonl"
        emb_path = memory_dir / "subtask" / "embeddings.npy"

        if segs_path.exists():
            self.segments = [
                SubtaskSegment.from_dict(json.loads(line))
                for line in segs_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        if emb_path.exists() and self.segments:
            self.embeddings = np.load(str(emb_path))
            if self.embeddings.ndim == 2 and len(self.embeddings) > 0:
                self._dim = self.embeddings.shape[1]

    def retrieve_for_subtasks(
        self,
        subtasks: list[str],
        min_similarity: float = 0.45,
    ) -> list[SubtaskSegment]:
        """Return the best-matching segment for each subtask label (deduplicated)."""
        if not self.segments or self.embeddings is None:
            return []
        seen: set[str] = set()
        results: list[SubtaskSegment] = []
        for sub in subtasks:
            q = _embed_query(sub, self._dim)
            sims = _cosine_sim(q, self.embeddings)
            best = int(np.argmax(sims))
            if sims[best] >= min_similarity:
                seg = self.segments[best]
                if seg.label not in seen:
                    seen.add(seg.label)
                    results.append(seg)
        return results

    @staticmethod
    def format_for_prompt(segments: list[SubtaskSegment]) -> str:
        if not segments:
            return ""
        lines = ["## Relevant Subtask Examples"]
        for seg in segments:
            lines.append(f"### {seg.label}")
            lines.append(seg.description)
            if seg.tool_calls:
                lines.append("Example calls:")
                for tc in seg.tool_calls:
                    args_str = json.dumps(tc.get("arguments", {}), ensure_ascii=False)
                    lines.append(f"  {tc['name']}({args_str})")
        return "\n".join(lines)


class FunctionMemoryStore:
    """Stores function-level records; retrieved by tool name for error-time hints."""

    def __init__(self, memory_dir: Path) -> None:
        self.records: list[FunctionStepRecord] = []
        self._by_tool: dict[str, list[FunctionStepRecord]] = {}
        self._dim: Optional[int] = None

        path = memory_dir / "function" / "records.jsonl"
        if path.exists():
            self.records = [
                FunctionStepRecord.from_dict(json.loads(line))
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        for rec in self.records:
            self._by_tool.setdefault(rec.tool_name, []).append(rec)

    def retrieve(
        self,
        tool_name: str,
        task_query: str = "",
        context: str = "",
        max_examples: int = 2,
    ) -> list[FunctionStepRecord]:
        candidates = self._by_tool.get(tool_name, [])
        if not candidates:
            return []
        if len(candidates) <= max_examples:
            return candidates[:max_examples]
        # Primary: rank by similarity between current agent context and stored think
        if context:
            with_think = [(i, r) for i, r in enumerate(candidates) if r.think]
            if with_think:
                q = _local_embed(context)
                stored_embs = np.array(
                    [_local_embed(r.think) for _, r in with_think], dtype=np.float32
                )
                sims = _cosine_sim(q, stored_embs)
                indices = np.argsort(sims)[::-1][:max_examples]
                return [with_think[i][1] for i in indices]
        # Fallback: rank by task_query similarity
        if task_query:
            q = _local_embed(task_query)
            stored_embs = np.array(
                [_local_embed(r.task_query) for r in candidates], dtype=np.float32
            )
            sims = _cosine_sim(q, stored_embs)
            indices = np.argsort(sims)[::-1][:max_examples]
            return [candidates[i] for i in indices]
        return candidates[:max_examples]

    @staticmethod
    def format_hint(tool_name: str, records: list[FunctionStepRecord]) -> str:
        if not records:
            return ""
        lines = [f"[Memory Hint] Past successful calls to `{tool_name}`:"]
        for rec in records:
            if rec.think:
                lines.append(f"[Why this function is called]\n{rec.think}")
            args_str = json.dumps(rec.arguments, ensure_ascii=False)
            lines.append(f"  {tool_name}({args_str})")
            lines.append(f"  → {rec.result}")
        return "\n".join(lines)
