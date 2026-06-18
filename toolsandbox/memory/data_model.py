# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Memory data models for teacher-student knowledge transfer."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class WorkflowMemory:
    """High-level workflow insight extracted from a successful teacher trajectory.

    Stored once per scenario; retrieved by semantic similarity to the task query.
    """

    scenario_name: str
    categories: list[str]
    query: str           # First user message – used as embedding key for retrieval
    insight: str         # 2-4 sentences describing the generalised workflow pattern
    involved_tools: list[str]
    source: str = "teacher"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WorkflowMemory:
        return cls(**d)


@dataclass
class SubtaskSegment:
    """A logical sub-step with masked example tool calls.

    Decomposed from a teacher trajectory by an LLM; retrieved by semantic
    similarity to decomposed subtask labels.
    """

    scenario_name: str
    label: str              # Short verb phrase, e.g. "look up contact phone number"
    description: str        # One-sentence description
    tool_calls: list[dict]  # [{name: str, arguments: dict}] – PII masked
    observations: list[str] # Sanitised tool results
    required_tools: list[str]
    query: str              # First user message – embedding key

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SubtaskSegment:
        return cls(**d)


@dataclass
class FunctionStepRecord:
    """A single successful tool call with masked arguments and result.

    Retrieved by tool name when the student encounters an execution error.
    """

    tool_name: str
    scenario_name: str
    step: int
    arguments: dict          # PII masked
    result: str              # Truncated and sanitised
    task_query: str          # First user message – used for similarity ranking
    think: str = ""          # LLM-generated explanation of why this function is called

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FunctionStepRecord:
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)
