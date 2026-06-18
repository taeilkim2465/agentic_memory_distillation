# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Build memory-augmented prompt strings for injection into agent messages."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from openai import OpenAI

from tool_sandbox.memory.store import (
    FunctionMemoryStore,
    SubtaskSegmentStore,
    WorkflowMemoryStore,
)

LOGGER = logging.getLogger(__name__)

_DECOMPOSE_SYSTEM = (
    "Break the given task into at most 6 logical subtasks. "
    "Return a JSON array (no markdown, no prose) of short string labels. "
    'Example: ["look up contact", "check cellular setting", "send message"]. '
    "Remove any <think>...</think> blocks before answering."
)

# Process-level cache: memory_dir → loaded stores
_STORE_CACHE: dict[Path, dict[str, object]] = {}


def _get_stores(memory_dir: Path) -> dict[str, object]:
    if memory_dir not in _STORE_CACHE:
        _STORE_CACHE[memory_dir] = {
            "workflow": WorkflowMemoryStore(memory_dir),
            "subtask": SubtaskSegmentStore(memory_dir),
            "function": FunctionMemoryStore(memory_dir),
        }
    return _STORE_CACHE[memory_dir]


def _decompose_task(query: str, decomposer_llm: str) -> list[str]:
    """Ask an LLM to decompose the task query into subtask labels."""
    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=decomposer_llm,
            messages=[
                {"role": "system", "content": _DECOMPOSE_SYSTEM},
                {"role": "user", "content": f"Task: {query}"},
            ],
            temperature=0.0,
        )
        raw = (resp.choices[0].message.content or "[]").strip()
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        result = json.loads(raw)
        if isinstance(result, list):
            return [str(x) for x in result]
    except Exception as exc:
        LOGGER.warning("Subtask decomposition failed: %s", exc)
    return []


def build_static_memory_prompt(
    query: str,
    memory_dir: Path,
    decomposer_llm: str = "gpt-4o-mini",
    enabled_types: Optional[set[str]] = None,
) -> str:
    """Return a <memory>…</memory> block to prepend to the system prompt.

    Retrieves workflow and subtask memories relevant to *query* from *memory_dir*.
    Returns an empty string when no useful memories are found.
    """
    if enabled_types is None:
        enabled_types = {"workflow", "subtask"}

    stores = _get_stores(memory_dir)
    parts: list[str] = []

    if "workflow" in enabled_types:
        wf_store: WorkflowMemoryStore = stores["workflow"]  # type: ignore[assignment]
        wf_memories = wf_store.retrieve(query, top_k=1, min_sim=0.5)
        formatted = wf_store.format_for_prompt(wf_memories)
        if formatted:
            parts.append(formatted)

    if "subtask" in enabled_types:
        sub_store: SubtaskSegmentStore = stores["subtask"]  # type: ignore[assignment]
        if sub_store.segments:  # skip LLM decomposition when store has no content
            subtasks = _decompose_task(query, decomposer_llm)
            if subtasks:
                segments = sub_store.retrieve_for_subtasks(subtasks)
                formatted = sub_store.format_for_prompt(segments)
                if formatted:
                    parts.append(formatted)

    if not parts:
        return ""
    return "<memory>\n" + "\n\n".join(parts) + "\n</memory>"


def build_error_hint(
    tool_name: str,
    error_msg: str,
    task_query: str,
    memory_dir: Path,
    context: str = "",
) -> Optional[str]:
    """Return a hint string when a tool call fails, drawn from past successes.

    Returns None when no relevant past calls are stored.
    """
    stores = _get_stores(memory_dir)
    fn_store: FunctionMemoryStore = stores["function"]  # type: ignore[assignment]
    records = fn_store.retrieve(tool_name, task_query=task_query, context=context, max_examples=2)
    if not records:
        return None
    hint_body = fn_store.format_hint(tool_name, records)
    return f"\nError during execution: {error_msg}\n\n{hint_body}"
