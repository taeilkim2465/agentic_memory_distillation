# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Extract structured trajectory from a ToolSandbox ExecutionContext."""

from __future__ import annotations

import ast
import json
import logging
from typing import Any, Optional

from tool_sandbox.common.execution_context import DatabaseNamespace, ExecutionContext, RoleType
from tool_sandbox.scenarios.user_simulator_few_shot_examples import USER_INSTRUCTION

LOGGER = logging.getLogger(__name__)

_USER_INSTRUCTION_STRIPPED = USER_INSTRUCTION.strip()


def extract_task_instruction(system_user_content: str) -> str:
    """Extract the task-specific part from a SYSTEM→USER message.

    SYSTEM→USER messages are formatted as USER_INSTRUCTION + task description.
    Returns just the task description, which is specific per scenario.
    """
    content = system_user_content.strip()
    if _USER_INSTRUCTION_STRIPPED in content:
        idx = content.index(_USER_INSTRUCTION_STRIPPED) + len(_USER_INSTRUCTION_STRIPPED)
        return content[idx:].strip()
    return content


def extract_trajectory(context: ExecutionContext) -> Optional[dict[str, Any]]:
    """Return a structured view of a completed scenario execution.

    Returns a dict with:
        query   – first USER->AGENT message (the task description)
        steps   – list of {tool_name, arguments, result, step_index}
        messages – list of {sender, recipient, content} in chronological order

    Returns None when no user query is found (e.g. empty conversation).
    """
    # get_all_history_snapshots=True is required for SANDBOX because each
    # message row lives at its own sandbox_message_index; the default filter
    # (== latest_index) would return only the final message.
    try:
        db = context.get_database(
            DatabaseNamespace.SANDBOX,
            get_all_history_snapshots=True,
            drop_sandbox_message_index=False,
            drop_headguard=True,
        )
    except Exception as exc:
        LOGGER.warning("Could not read SANDBOX database: %s", exc)
        return None

    if db.is_empty():
        return None

    rows = db.sort("sandbox_message_index").to_dicts()

    task_instruction: Optional[str] = None  # from SYSTEM→USER (specific per scenario)
    query: Optional[str] = None             # fallback: first USER→AGENT message
    messages: list[dict[str, Any]] = []
    # Maps tool_call_id -> {name, arguments, step_index} while waiting for result
    pending: dict[str, dict[str, Any]] = {}
    steps: list[dict[str, Any]] = []
    step_counter = 0

    for row in rows:
        sender: str = row["sender"]
        recipient: str = row["recipient"]
        content: str = row.get("content") or ""
        fn_name: str = row.get("openai_function_name") or ""
        call_id: str = row.get("openai_tool_call_id") or ""

        messages.append({"sender": sender, "recipient": recipient, "content": content})

        # Prefer SYSTEM→USER message: contains the actual task description.
        # Use the LAST such message (earlier ones are few-shot examples).
        if sender == RoleType.SYSTEM and recipient == RoleType.USER:
            task_instruction = extract_task_instruction(content)

        # Fallback: first USER→AGENT message
        if query is None and sender == RoleType.USER and recipient == RoleType.AGENT:
            query = content

        # Agent tool call → EXECUTION_ENVIRONMENT
        if (
            sender == RoleType.AGENT
            and recipient == RoleType.EXECUTION_ENVIRONMENT
            and fn_name
        ):
            args = _parse_arguments_from_code(content)
            key = call_id or fn_name
            pending[key] = {
                "name": fn_name,
                "arguments": args,
                "step_index": step_counter,
            }
            step_counter += 1

        # Tool result → Agent
        if sender == RoleType.EXECUTION_ENVIRONMENT and recipient == RoleType.AGENT:
            # Match by call_id first, then fall back to oldest pending entry
            key = call_id if call_id in pending else (next(iter(pending), None))
            if key and key in pending:
                call = pending.pop(key)
                steps.append(
                    {
                        "tool_name": call["name"],
                        "arguments": call["arguments"],
                        "result": content,
                        "step_index": call["step_index"],
                    }
                )

    effective_query = task_instruction or query
    if effective_query is None:
        return None

    return {"query": effective_query, "steps": steps, "messages": messages}


def _parse_arguments_from_code(code: str) -> dict[str, Any]:
    """Extract the argument dict from generated Python execution code.

    The code format produced by openai_tool_call_to_python_code is:
        {tool_id}_parameters = {<dict literal>}
        {tool_id}_response = func_name(**{tool_id}_parameters)
        print(repr({tool_id}_response))
    """
    for line in code.splitlines():
        line = line.strip()
        if "_parameters = " in line:
            _, _, rhs = line.partition(" = ")
            try:
                result = ast.literal_eval(rhs.strip())
                if isinstance(result, dict):
                    return result
            except Exception:
                # Try JSON as a fallback
                try:
                    return json.loads(rhs.strip())
                except Exception:
                    pass
    return {}
