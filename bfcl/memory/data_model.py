"""
BFCL memory data model.
Adapted from tau2-bench for function-call leaderboard trajectories.
"""
import json
import re
import zlib
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class WorkflowMemDoc:
    """Workflow-level memory: generalized insight from a completed teacher trajectory."""

    task_id: str
    category: str        # BFCL test category (e.g. "multi_turn_base")
    query: str           # first user message — used for retrieval
    insight: str         # 2-4 sentence generalized workflow insight
    involved_classes: list[str] = field(default_factory=list)
    source: str = "teacher"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowMemDoc":
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


@dataclass
class SubtaskSegment:
    """Subtask-level memory: tool call examples extracted from teacher trajectories."""

    task_id: str
    category: str
    label: str           # short action label (e.g. "create directory")
    description: str     # one-sentence description
    tool_calls: list[str] = field(default_factory=list)    # masked calls (placeholder args)
    observations: list[str] = field(default_factory=list)  # sanitized results from trajectory
    required_tools: list[str] = field(default_factory=list)
    trajectory_excerpt: str = ""   # sanitized step output from teacher
    source: str = "teacher"
    source_task_id: str = ""
    generated_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubtaskSegment":
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


@dataclass
class FunctionStepRecord:
    """Per-step tool call record extracted from teacher trajectory."""

    tool_name: str
    category: str
    task_id: str
    step: int
    arguments: dict[str, Any] = field(default_factory=dict)
    result: str = ""           # truncated result (~500 chars)
    task_query: str = ""       # first user message of the task
    turn_instruction: str = "" # user instruction of the specific turn where this call occurs
    think: str = ""            # LLM-generated explanation of why this function is called
    generated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FunctionStepRecord":
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


def sanitize_memory_text(text: str) -> str:
    """Remove task-specific values to make memory reusable."""
    s = str(text or "")
    s = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "<EMAIL>", s)
    s = re.sub(r"\+?\d[\d\-\s()]{7,}\d", "<PHONE>", s)
    s = re.sub(r"\$\s*\d+(?:\.\d+)?", "<AMOUNT>", s)
    s = re.sub(r"\b\d{6,}\b", "<NUM>", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def local_embedding(text: str, dim: int = 256) -> list[float]:
    """Deterministic bag-of-words fallback embedding (no API needed)."""
    import numpy as np
    tokens = re.findall(r"[a-z0-9_]+", (text or "").lower())
    vec = np.zeros(dim, dtype=np.float32)
    for token in tokens:
        vec[zlib.crc32(token.encode()) % dim] += 1.0
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec.tolist()
