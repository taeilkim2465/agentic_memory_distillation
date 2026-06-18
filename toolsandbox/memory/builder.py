# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Build and persist memory entries from successful teacher trajectories."""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
from openai import OpenAI

from tool_sandbox.common.execution_context import ExecutionContext
from tool_sandbox.memory.data_model import (
    FunctionStepRecord,
    SubtaskSegment,
    WorkflowMemory,
)
from tool_sandbox.memory.sanitize import mask_arguments, sanitize_result
from tool_sandbox.memory.trajectory import extract_trajectory

LOGGER = logging.getLogger(__name__)

_WORKFLOW_SYSTEM = (
    "You generate concise, reusable memory insights from successful tool-calling "
    "trajectories. Write 2-4 sentences capturing the general workflow pattern: "
    "which tools to call and in what order, and what to watch out for. "
    "Replace all concrete values (names, phone numbers, emails, IDs) with "
    "placeholders such as <NAME>, <PHONE>, <EMAIL>, <ID>. "
    "Keep it under 150 words."
)

_WORKFLOW_SYSTEM_NO_TOOLS = (
    "You generate concise, reusable memory insights from successful agent trajectories. "
    "This trajectory has no tool calls: the agent correctly recognized that it lacked "
    "sufficient information to complete the task and informed the user instead of calling tools. "
    "Write 2-4 sentences describing when this task type requires additional information, "
    "what information is missing, and that the correct behavior is to tell the user "
    "what is needed rather than attempting to call tools. "
    "Replace all concrete values with placeholders such as <NAME>, <PHONE>, <EMAIL>, <ID>. "
    "Keep it under 150 words."
)

_SUBTASK_SYSTEM = (
    "You analyse a successful tool-calling trajectory and decompose it into logical "
    "subtasks. Return a JSON array (no markdown, no prose) of at most 6 objects, "
    "each with exactly these keys: "
    '{"label": "short verb phrase", "description": "one sentence", '
    '"required_tools": ["tool_name", ...]}. '
    "Labels should be short verb phrases like 'look up contact' or 'check cellular setting'."
)

_THINK_SYSTEM = "You generate concise reasoning for function-calling agents."

_THINK_USER_TEMPLATE = """\
A function-calling agent is solving the following task:
Task: {task_query}

{prev_steps_section}\
Next action: {tool_name}({args_str})

In 2-3 sentences:
1. Why this function is being called and what it provides for the task
2. What prerequisite information was already available from prior calls
3. Key constraints on the arguments to keep in mind

Reply with only the reasoning, no bullets, no markdown.
"""


class ToolSandboxMemoryBuilder:
    """Builds workflow, subtask, and function memories from teacher scenario results."""

    def __init__(
        self,
        memory_dir: Path,
        builder_llm: str = "gpt-5-mini",
        embedding_model: str = "text-embedding-3-small",
        think_model: Optional[str] = None,
    ) -> None:
        self.memory_dir = Path(memory_dir)
        self.builder_llm = builder_llm
        self.embedding_model = embedding_model
        self.think_model = think_model
        self._client = OpenAI()
        # Embedding client always targets the real OpenAI API, not a vllm server
        import os
        embed_base_url = os.environ.get("OPENAI_EMBED_BASE_URL", "https://api.openai.com/v1")
        self._embed_client = OpenAI(base_url=embed_base_url)

    # ------------------------------------------------------------------
    # Public API

    def build_from_scenario_result(
        self,
        scenario_name: str,
        categories: list[str],
        context: ExecutionContext,
    ) -> dict[str, int]:
        """Extract memories from a successful scenario execution and persist them.

        Returns a dict with counts of each memory type saved.
        """
        trajectory = extract_trajectory(context)
        if not trajectory:
            LOGGER.warning("No usable trajectory found for scenario '%s'", scenario_name)
            return {"workflow": 0, "subtask": 0, "function": 0}

        counts: dict[str, int] = {}

        try:
            wf = self._build_workflow(scenario_name, categories, trajectory)
            self._persist_workflow(wf)
            counts["workflow"] = 1
        except Exception as exc:
            LOGGER.error("Workflow build failed for '%s': %s", scenario_name, exc)
            counts["workflow"] = 0

        # subtask and function memory require tool call steps
        if trajectory["steps"]:
            try:
                segs = self._build_subtasks(scenario_name, trajectory)
                self._persist_subtasks(segs)
                counts["subtask"] = len(segs)
            except Exception as exc:
                LOGGER.error("Subtask build failed for '%s': %s", scenario_name, exc)
                counts["subtask"] = 0

            try:
                records = self._build_function_records(scenario_name, trajectory)
                self._persist_function_records(records)
                counts["function"] = len(records)
            except Exception as exc:
                LOGGER.error("Function records build failed for '%s': %s", scenario_name, exc)
                counts["function"] = 0
        else:
            counts["subtask"] = 0
            counts["function"] = 0

        return counts

    # ------------------------------------------------------------------
    # Builders (call LLM to generate memories)

    def _build_workflow(
        self,
        scenario_name: str,
        categories: list[str],
        trajectory: dict[str, Any],
    ) -> WorkflowMemory:
        has_steps = bool(trajectory["steps"])
        system_prompt = _WORKFLOW_SYSTEM if has_steps else _WORKFLOW_SYSTEM_NO_TOOLS
        prompt = _format_trajectory(trajectory)
        resp = self._client.chat.completions.create(
            model=self.builder_llm,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Trajectory:\n{prompt}\n\nGenerate the workflow memory insight:",
                },
            ],
            temperature=1,
            max_tokens=1024,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        insight = re.sub(r"<think>.*?</think>\s*", "", resp.choices[0].message.content or "", flags=re.DOTALL).strip()
        involved_tools = list(dict.fromkeys(s["tool_name"] for s in trajectory["steps"]))
        return WorkflowMemory(
            scenario_name=scenario_name,
            categories=categories,
            query=trajectory["query"],
            insight=insight,
            involved_tools=involved_tools,
        )

    def _build_subtasks(
        self,
        scenario_name: str,
        trajectory: dict[str, Any],
    ) -> list[SubtaskSegment]:
        prompt = _format_trajectory(trajectory)
        resp = self._client.chat.completions.create(
            model=self.builder_llm,
            messages=[
                {"role": "system", "content": _SUBTASK_SYSTEM},
                {
                    "role": "user",
                    "content": f"Trajectory:\n{prompt}\n\nDecompose into subtasks:",
                },
            ],
            temperature=1,
            max_tokens=1024,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw = (resp.choices[0].message.content or "[]").strip()
        items: list[dict[str, Any]] = _parse_json_list(raw)

        # Build per-tool lookup for fast annotation
        steps_by_tool: dict[str, list[dict[str, Any]]] = {}
        for step in trajectory["steps"]:
            steps_by_tool.setdefault(step["tool_name"], []).append(step)

        segments: list[SubtaskSegment] = []
        for item in items:
            required: list[str] = item.get("required_tools", [])
            tool_calls: list[dict[str, Any]] = []
            observations: list[str] = []
            for tool_name in required:
                for step in steps_by_tool.get(tool_name, []):
                    tool_calls.append(
                        {
                            "name": tool_name,
                            "arguments": mask_arguments(step["arguments"]),
                        }
                    )
                    observations.append(sanitize_result(step["result"]))
            segments.append(
                SubtaskSegment(
                    scenario_name=scenario_name,
                    label=item.get("label", ""),
                    description=item.get("description", ""),
                    tool_calls=tool_calls,
                    observations=observations,
                    required_tools=required,
                    query=trajectory["query"],
                )
            )
        return segments

    def _build_function_records(
        self,
        scenario_name: str,
        trajectory: dict[str, Any],
    ) -> list[FunctionStepRecord]:
        import re
        _error_re = re.compile(r"Error|Exception|Traceback|failed", re.IGNORECASE)

        records = []
        steps = trajectory["steps"]
        task_query = trajectory["query"]
        for i, step in enumerate(steps):
            if _error_re.search(str(step.get("result", ""))):
                continue
            think = self._generate_think(task_query, step, steps[:i]) if self.think_model else ""
            records.append(
                FunctionStepRecord(
                    tool_name=step["tool_name"],
                    scenario_name=scenario_name,
                    step=step["step_index"],
                    arguments=mask_arguments(step["arguments"]),
                    result=sanitize_result(step["result"]),
                    task_query=task_query,
                    think=think,
                )
            )
        return records

    def _generate_think(
        self,
        task_query: str,
        current_step: dict[str, Any],
        prev_steps: list[dict[str, Any]],
    ) -> str:
        import re

        prev_steps_section = ""
        if prev_steps:
            lines = ["Previous calls:"]
            for s in prev_steps:
                args_str = json.dumps(mask_arguments(s["arguments"]), ensure_ascii=False)
                lines.append(f"  {s['tool_name']}({args_str}) → {sanitize_result(s['result'], max_len=100)}")
            prev_steps_section = "\n".join(lines) + "\n\n"

        args_str = json.dumps(mask_arguments(current_step["arguments"]), ensure_ascii=False)
        prompt = _THINK_USER_TEMPLATE.format(
            task_query=task_query,
            prev_steps_section=prev_steps_section,
            tool_name=current_step["tool_name"],
            args_str=args_str,
        )
        try:
            resp = self._client.chat.completions.create(
                model=self.think_model,
                messages=[
                    {"role": "system", "content": _THINK_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=1,
                max_tokens=1024,
            )
            content = (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            LOGGER.warning("Think generation failed for '%s': %s", current_step["tool_name"], exc)
            return ""

        content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL).strip()
        return content

    # ------------------------------------------------------------------
    # Persistence (file-lock protected for parallel runs)

    @contextlib.contextmanager
    def _lock(self):
        lock_path = self.memory_dir / ".write.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

    def _persist_workflow(self, memory: WorkflowMemory) -> None:
        wf_dir = self.memory_dir / "workflow"
        wf_dir.mkdir(parents=True, exist_ok=True)
        docs_path = wf_dir / "documents.json"
        emb_path = wf_dir / "embeddings.npy"

        new_emb = self._embed(memory.query).reshape(1, -1)
        with self._lock():
            docs: list[dict[str, Any]] = []
            if docs_path.exists():
                docs = json.loads(docs_path.read_text(encoding="utf-8"))
            docs.append(memory.to_dict())
            docs_path.write_text(
                json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            if emb_path.exists():
                existing = np.load(str(emb_path))
                if existing.shape[1] == new_emb.shape[1]:
                    new_emb = np.vstack([existing, new_emb])
            np.save(str(emb_path), new_emb)

    def _persist_subtasks(self, segments: list[SubtaskSegment]) -> None:
        if not segments:
            return
        sub_dir = self.memory_dir / "subtask"
        sub_dir.mkdir(parents=True, exist_ok=True)
        segs_path = sub_dir / "segments.jsonl"
        emb_path = sub_dir / "embeddings.npy"

        new_embs = np.array(
            [self._embed(seg.label + ": " + seg.description) for seg in segments],
            dtype=np.float32,
        )
        with self._lock():
            with open(segs_path, "a", encoding="utf-8") as f:
                for seg in segments:
                    f.write(json.dumps(seg.to_dict(), ensure_ascii=False) + "\n")

            if emb_path.exists():
                existing = np.load(str(emb_path))
                if existing.shape[1] == new_embs.shape[1]:
                    new_embs = np.vstack([existing, new_embs])
            np.save(str(emb_path), new_embs)

    def _persist_function_records(self, records: list[FunctionStepRecord]) -> None:
        if not records:
            return
        fn_dir = self.memory_dir / "function"
        fn_dir.mkdir(parents=True, exist_ok=True)
        path = fn_dir / "records.jsonl"
        with self._lock():
            with open(path, "a", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Embedding

    def _embed(self, text: str) -> np.ndarray:
        try:
            resp = self._embed_client.embeddings.create(
                model=self.embedding_model, input=text
            )
            return np.array(resp.data[0].embedding, dtype=np.float32)
        except Exception:
            return _local_embed(text)


# ------------------------------------------------------------------
# Helpers

def _format_trajectory(trajectory: dict[str, Any]) -> str:
    """Format a trajectory dict as a readable prompt string."""
    lines = [f"Task: {trajectory['query']}", ""]
    for i, step in enumerate(trajectory["steps"], 1):
        args_str = json.dumps(mask_arguments(step["arguments"]), ensure_ascii=False)
        result_str = sanitize_result(step["result"])
        lines.append(f"Step {i}: {step['tool_name']}({args_str})")
        lines.append(f"  → {result_str}")
    return "\n".join(lines)


def _parse_json_list(raw: str) -> list[dict[str, Any]]:
    """Parse a JSON array from LLM output, tolerating surrounding prose."""
    import re

    # Strip complete <think>...</think> blocks
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    # Strip unclosed <think> blocks (truncated by max_tokens)
    raw = re.sub(r"<think>.*$", "", raw, flags=re.DOTALL).strip()
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    return []


def _local_embed(text: str, dim: int = 256) -> np.ndarray:
    """Deterministic character n-gram embedding as a fallback when the API is unavailable."""
    vec = np.zeros(dim, dtype=np.float32)
    text = text.lower()
    for n in (2, 3):
        for i in range(len(text) - n + 1):
            vec[hash(text[i : i + n]) % dim] += 1.0
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec
