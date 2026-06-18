# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Agent role augmented with teacher-derived memory retrieval."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Union

from openai import NotGiven
from openai.types.chat import ChatCompletion, ChatCompletionToolParam

from tool_sandbox.common.execution_context import RoleType
from tool_sandbox.memory.injection import build_error_hint, build_static_memory_prompt
from tool_sandbox.memory.trajectory import extract_task_instruction
from tool_sandbox.roles.openai_api_agent import OpenAICompatibleServerAgent

LOGGER = logging.getLogger(__name__)

_ERROR_RE = re.compile(r"Error|Exception|Traceback|failed", re.IGNORECASE)


class MemoryAugmentedAgent(OpenAICompatibleServerAgent):
    """OpenAI-compatible agent that augments its prompts with teacher memories.

    **Static memory** (injected once, before the first LLM call):
        Workflow insight + subtask examples retrieved from the memory store by
        semantic similarity to the task query.

    **Dynamic error hints** (injected after each failed tool call):
        Past successful calls to the same tool, drawn from the function memory store.
    """

    def __init__(
        self,
        model_name: str,
        memory_dir: Path,
        enabled_types: Optional[set[str]] = None,
        use_function_hints: bool = True,
        max_tokens: Optional[int] = None,
    ) -> None:
        super().__init__(model_name=model_name)
        self.memory_dir = Path(memory_dir)
        self.decomposer_llm = model_name
        self.max_tokens = max_tokens
        # None → use injection.py defaults (workflow + subtask)
        self.enabled_types = enabled_types
        self.use_function_hints = use_function_hints
        self._first_user_query: Optional[str] = None
        self._static_memory: Optional[str] = None   # "" = computed but empty
        self._memory_initialized: bool = False
        self.memory_log: list[dict] = []

    # ------------------------------------------------------------------
    # BaseRole interface

    def respond(self, ending_index: Optional[int] = None) -> None:
        # Capture task query for memory retrieval.
        # Prefer SYSTEM→USER message (contains the actual task description, scenario-specific).
        # Fall back to first USER→AGENT message.
        if self._first_user_query is None:
            messages = self.get_messages(ending_index=ending_index)
            for msg in messages:
                if msg.sender == RoleType.SYSTEM and msg.recipient == RoleType.USER:
                    task_part = extract_task_instruction(msg.content)
                    if task_part:
                        self._first_user_query = task_part
                        # Don't break: keep overwriting to get the LAST SYSTEM→USER
                        # (earlier ones are few-shot examples)
            if self._first_user_query is None:
                filtered = self.filter_messages(messages)
                for msg in filtered:
                    if msg.sender == RoleType.USER and msg.recipient == RoleType.AGENT:
                        self._first_user_query = msg.content
                        break
        super().respond(ending_index=ending_index)

    def reset(self) -> None:
        super().reset()
        self._first_user_query = None
        self._static_memory = None
        self._memory_initialized = False
        self.memory_log = []

    # ------------------------------------------------------------------
    # OpenAIAPIAgent interface

    def model_inference(
        self,
        openai_messages: list[
            dict[Literal["role", "content", "tool_call_id", "name", "tool_calls"], Any]
        ],
        openai_tools: Union[Iterable[ChatCompletionToolParam], NotGiven],
    ) -> ChatCompletion:
        augmented = list(openai_messages)

        # ── Static memory (once per scenario) ──────────────────────────
        if not self._memory_initialized and self._first_user_query:
            try:
                self._static_memory = build_static_memory_prompt(
                    query=self._first_user_query,
                    memory_dir=self.memory_dir,
                    decomposer_llm=self.decomposer_llm,
                    enabled_types=self.enabled_types,
                )
            except Exception as exc:
                LOGGER.warning("Static memory retrieval failed: %s", exc)
                self._static_memory = ""
            self._memory_initialized = True
            # Log only once when memory is first computed
            self.memory_log.append({
                "type": "static",
                "query": self._first_user_query,
                "content": self._static_memory if self._static_memory else None,
            })

        if self._static_memory:
            if augmented and augmented[0]["role"] == "system":
                augmented[0] = {
                    **augmented[0],
                    "content": self._static_memory + "\n\n" + augmented[0]["content"],
                }
            else:
                augmented.insert(0, {"role": "system", "content": self._static_memory})

        # ── Dynamic error hints ─────────────────────────────────────────
        if self.use_function_hints:
            augmented = self._inject_error_hints(augmented, openai_tools)

        return super().model_inference(augmented, openai_tools)

    # ------------------------------------------------------------------
    # Private helpers

    def _inject_error_hints(
        self,
        messages: list[dict[str, Any]],
        openai_tools: Union[Iterable[ChatCompletionToolParam], NotGiven],
    ) -> list[dict[str, Any]]:
        """Append a function-memory hint to the most recent failed tool result."""
        if not self._first_user_query:
            return messages

        tool_schema_map = _build_tool_schema_map(openai_tools)

        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            content: str = msg.get("content") or ""
            if msg.get("role") == "tool" and _ERROR_RE.search(content):
                tool_name = _extract_tool_name(messages, i)
                if tool_name:
                    context = _build_call_context(messages, i)
                    try:
                        hint = build_error_hint(
                            tool_name=tool_name,
                            error_msg=content[:300],
                            task_query=self._first_user_query,
                            memory_dir=self.memory_dir,
                            context=context,
                        )
                    except Exception as exc:
                        LOGGER.warning("Error hint retrieval failed: %s", exc)
                        hint = None

                    func_desc = _format_func_description(tool_schema_map.get(tool_name))
                    if hint or func_desc:
                        extra = ""
                        if func_desc:
                            extra += f"\n{func_desc}"
                        if hint:
                            extra += f"\n{hint}"
                        messages = list(messages)
                        messages[i] = {**msg, "content": content + extra}
                        self.memory_log.append({
                            "type": "error_hint",
                            "tool_name": tool_name,
                            "error_msg": content[:300],
                            "hint": extra,
                        })
                break  # Only augment the most recent error

        return messages


def _extract_tool_name(messages: list[dict[str, Any]], tool_msg_idx: int) -> Optional[str]:
    """Find the function name for a tool-result message by matching its tool_call_id."""
    target_id = messages[tool_msg_idx].get("tool_call_id")
    for i in range(tool_msg_idx - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                if tc.get("id") == target_id:
                    return tc.get("function", {}).get("name")
    return None


def _build_call_context(messages: list[dict[str, Any]], error_msg_idx: int) -> str:
    """Build a context string from recent tool calls preceding the failed call.

    Combines the previous assistant tool calls (up to 3) and the current
    failed call so it can be matched against stored think strings.
    """
    target_id = messages[error_msg_idx].get("tool_call_id")
    prev_calls: list[str] = []
    current_call: str = ""

    for i in range(error_msg_idx - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            name = tc.get("function", {}).get("name", "")
            args = tc.get("function", {}).get("arguments", "{}")
            call_str = f"{name}({args})"
            if tc.get("id") == target_id:
                current_call = call_str
            else:
                prev_calls.append(call_str)
        if prev_calls or current_call:
            break

    parts: list[str] = []
    if prev_calls:
        parts.append("Previous: " + "; ".join(reversed(prev_calls[-3:])))
    if current_call:
        parts.append("Current: " + current_call)
    return " | ".join(parts)


def _build_tool_schema_map(
    openai_tools: Union[Iterable[ChatCompletionToolParam], NotGiven],
) -> dict[str, dict]:
    """Return a {tool_name: function_schema} map from openai_tools."""
    if isinstance(openai_tools, NotGiven) or openai_tools is None:
        return {}
    result = {}
    for tool in openai_tools:
        if isinstance(tool, dict):
            fn = tool.get("function", {})
        else:
            fn = getattr(tool, "function", {}) or {}
            if not isinstance(fn, dict):
                fn = fn.__dict__ if hasattr(fn, "__dict__") else {}
        name = fn.get("name", "")
        if name:
            result[name] = fn
    return result


def _format_func_description(fn_schema: Optional[dict]) -> str:
    """Format an OpenAI function schema into a concise description string."""
    if not fn_schema:
        return ""
    name = fn_schema.get("name", "")
    desc = fn_schema.get("description", "")
    params = fn_schema.get("parameters", {})
    props = params.get("properties", {})
    required = set(params.get("required", []))

    lines = [f"[Function Description] {name}: {desc}"]
    if props:
        lines.append("Parameters:")
        for param, info in props.items():
            req_mark = " (required)" if param in required else ""
            ptype = info.get("type", "")
            pdesc = info.get("description", "")
            lines.append(f"  - {param} ({ptype}){req_mark}: {pdesc}")
    return "\n".join(lines)
