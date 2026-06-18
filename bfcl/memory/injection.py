"""
Runtime memory injection for BFCL inference.

- Static injection: workflow + subtask memory into the system prompt before the first turn.
- Dynamic injection: function hints appended to error execution results.

Stores are loaded lazily and cached per memory_dir path to avoid repeated disk I/O.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional, Set

_C_RESET  = "\033[0m"
_C_BOLD   = "\033[1m"
_C_MAGENTA = "\033[95m"
_C_CYAN   = "\033[96m"
_C_YELLOW = "\033[93m"

_STORE_CACHE: dict[str, dict] = {}

_SUBTASK_DECOMPOSE_PROMPT = """\
Break the given multi-turn function-calling task into sequential subtasks.
Each subtask = a short action phrase (verb + object) representing one logical operation (one or a few related tool calls).
The task has multiple turns separated by " | ". Decompose the full sequence of turns together.

Available tools: {tool_names}

Rules:
- Each subtask = a short action phrase describing the tool operation (verb + object).
- Order subtasks by the sequence in which tool calls must occur across all turns.
- Reply with ONLY a JSON array of strings. At most {max_subtasks} subtasks.

Examples:

Task (multi-turn): Pop on over to the 'Documents' directory and craft a new file dubbed 'summary.txt' | In 'Documents', let's capture some profound topic 'quantum computing' | Count words in summary.txt
["navigate to directory", "create file", "write content to file", "count words"]

Task (multi-turn): Can you provide the latest trading details for Quasar Ltd.? | Please compile a comprehensive list of stock symbols in the technology sector.
["get latest stock details", "add tech stocks to watchlist"]

Task (multi-turn): Update the market status with the current time. | Purchase 100 Apple shares at prevailing market price. | Review the AAPL order details and account info. | Cancel the order and confirm.
["resolve symbol and time", "place buy order", "review order and account", "cancel order and confirm"]

Task (multi-turn): {instruction}
"""


def _get_stores(memory_dir: Path) -> dict:
    key = str(memory_dir)
    if key not in _STORE_CACHE:
        from bfcl_eval.memory.store import (
            FunctionMemoryStore,
            SubtaskSegmentStore,
            WorkflowMemoryStore,
        )
        _STORE_CACHE[key] = {
            "wf": WorkflowMemoryStore(memory_dir),
            "seg": SubtaskSegmentStore(memory_dir),
            "func": FunctionMemoryStore(memory_dir),
        }
    return _STORE_CACHE[key]


def _decompose_task(
    instruction: str,
    tool_names: list[str],
    decomposer_llm: str,
    max_subtasks: int = 6,
) -> list[str]:
    prompt = _SUBTASK_DECOMPOSE_PROMPT.format(
        tool_names=", ".join(tool_names),
        max_subtasks=max_subtasks,
        instruction=instruction,
    )
    try:
        import os
        import litellm
        kwargs: dict = {}
        if decomposer_llm.startswith("openai/"):
            base_url = os.getenv("VLLM_BASE_URL", "")
            if base_url:
                kwargs["api_base"] = base_url
                kwargs["api_key"] = os.getenv("VLLM_API_KEY", "EMPTY")
        resp = litellm.completion(
            model=decomposer_llm,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            timeout=30.0,
            **kwargs,
        )
        content = resp.choices[0].message.content or ""
    except Exception:
        return []

    content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL).strip()
    m = re.search(r"\[.*?\]", content, re.DOTALL)
    if not m:
        return []
    try:
        subtasks = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [s for s in subtasks if isinstance(s, str) and s.strip()]


def _extract_first_user_message(test_entry: dict) -> str:
    for turn in test_entry.get("question", []):
        for msg in turn:
            if msg.get("role") == "user":
                return msg.get("content", "")
    return test_entry.get("id", "")


def _extract_all_user_messages(test_entry: dict) -> str:
    """Concatenate all user instructions across turns for better task coverage."""
    parts = []
    for turn in test_entry.get("question", []):
        for msg in turn:
            if msg.get("role") == "user":
                parts.append(msg.get("content", ""))
                break
    return " | ".join(parts) if parts else test_entry.get("id", "")


def _extract_turn_instructions(test_entry: dict) -> list[str]:
    """Return each turn's user instruction as a separate subtask (no decomposer LLM needed)."""
    parts = []
    for turn in test_entry.get("question", []):
        for msg in turn:
            if msg.get("role") == "user":
                parts.append(msg.get("content", ""))
                break
    return parts


def _extract_tool_names(test_entry: dict) -> list[str]:
    return [f.get("name", "") for f in test_entry.get("function", []) if f.get("name")]


def _extract_func_name_from_call(func_call: str) -> str:
    """Extract bare function name from a decoded call string like 'mkdir(path=...)' or 'obj.mkdir(...)'."""
    m = re.match(r"[\w.]+", func_call)
    if not m:
        return ""
    name = m.group(0)
    return name.split(".")[-1]


def inject_static_memory(
    test_entry: dict,
    memory_dir: Path,
    decomposer_llm: str = "gpt-4o-mini",
    top_k_workflow: int = 1,
    enabled_types: Optional[set] = None,
    subtask_source: str = "turns",  # "turns" | "decompose"
    max_memory_chars: int = 6000,  # guard against context overflow
) -> dict:
    """
    Prepend workflow + subtask memory to the system prompt of test_entry["question"].
    Modifies test_entry in-place and returns it.

    subtask_source:
      "turns"     — use each turn's user instruction directly as subtasks (default, no LLM call)
      "decompose" — decompose full task with decomposer_llm
    """
    if enabled_types is None:
        enabled_types = {"workflow", "subtask"}

    stores = _get_stores(memory_dir)
    wf_store = stores["wf"]
    seg_store = stores["seg"]

    query = _extract_first_user_message(test_entry)
    category = test_entry.get("id", "").rsplit("_", 1)[0]

    memory_parts: list[str] = []

    # --- Workflow memory ---
    if "workflow" in enabled_types:
        wf_docs = wf_store.retrieve(query, top_k=top_k_workflow, category=category)
        if not wf_docs:
            wf_docs = wf_store.retrieve(query, top_k=top_k_workflow)
        if wf_docs:
            wf_text = wf_store.format_for_prompt(wf_docs)
            memory_parts.append(f"## Workflow Memory (similar past tasks)\n{wf_text}")

    # --- Subtask memory ---
    if "subtask" in enabled_types:
        tool_names = _extract_tool_names(test_entry)
        if subtask_source == "turns":
            subtasks = _extract_turn_instructions(test_entry)
        else:
            full_query = _extract_all_user_messages(test_entry)
            subtasks = _decompose_task(full_query, tool_names, decomposer_llm)
        if subtasks and len(seg_store) > 0:
            segs = seg_store.retrieve_for_subtasks(subtasks, category=category)
            if not segs:
                segs = seg_store.retrieve_for_subtasks(subtasks)
            if segs:
                seg_text = seg_store.format_for_prompt(segs)
                memory_parts.append(f"## Subtask Memory (tool call examples)\n{seg_text}")

    if not memory_parts:
        return test_entry

    memory_block = "\n\n".join(memory_parts)
    if len(memory_block) > max_memory_chars:
        memory_block = memory_block[:max_memory_chars] + "\n... (truncated)"

    injection = (
        f"<memory>\n{memory_block}\n</memory>\n\n"
        "Use the memory above as reference when solving the task. "
        "Adapt it to the current context rather than copying values literally."
    )

    task_id = test_entry.get("id", "?")
    print(
        f"\n{_C_BOLD}{_C_MAGENTA}{'━' * 80}{_C_RESET}\n"
        f"{_C_BOLD}{_C_MAGENTA}[MEMORY INJECT] {task_id}{_C_RESET}\n"
        f"{_C_CYAN}{injection}{_C_RESET}\n"
        f"{_C_BOLD}{_C_MAGENTA}{'━' * 80}{_C_RESET}",
        flush=True,
    )

    first_turn = test_entry["question"][0]
    if first_turn and first_turn[0].get("role") == "system":
        first_turn[0]["content"] = injection + "\n\n" + first_turn[0]["content"]
    else:
        first_turn.insert(0, {"role": "system", "content": injection})

    return test_entry


def get_subtask_text_for_turn(
    turn_instruction: str,
    test_entry: dict,
    memory_dir: Path,
    max_chars: int = 2000,
) -> str:
    """Return formatted subtask memory text for a single turn instruction, or empty string."""
    stores = _get_stores(memory_dir)
    seg_store = stores["seg"]
    if len(seg_store) == 0:
        return ""
    category = test_entry.get("id", "").rsplit("_", 1)[0]
    segs = seg_store.retrieve_for_subtasks([turn_instruction], category=category)
    if not segs:
        segs = seg_store.retrieve_for_subtasks([turn_instruction])
    if not segs:
        return ""
    text = seg_store.format_for_prompt(segs)
    return text[:max_chars] + "\n... (truncated)" if len(text) > max_chars else text


def inject_dynamic_memory_for_turn(
    turn_instruction: str,
    test_entry: dict,
    memory_dir: Path,
    enabled_types: Optional[set] = None,
    max_memory_chars: int = 3000,
) -> str:
    """
    Retrieve memory relevant to a single turn's instruction and return the injection string.
    Used for per-turn dynamic retrieval (as opposed to static injection before Turn 1).
    Returns empty string if nothing is found.
    """
    if enabled_types is None:
        enabled_types = {"subtask"}

    stores = _get_stores(memory_dir)
    wf_store = stores["wf"]
    seg_store = stores["seg"]
    category = test_entry.get("id", "").rsplit("_", 1)[0]

    memory_parts: list[str] = []

    if "workflow" in enabled_types:
        wf_docs = wf_store.retrieve(turn_instruction, top_k=1, category=category)
        if not wf_docs:
            wf_docs = wf_store.retrieve(turn_instruction, top_k=1)
        if wf_docs:
            memory_parts.append(f"## Workflow Memory\n{wf_store.format_for_prompt(wf_docs)}")

    if "subtask" in enabled_types and len(seg_store) > 0:
        segs = seg_store.retrieve_for_subtasks([turn_instruction], category=category)
        if not segs:
            segs = seg_store.retrieve_for_subtasks([turn_instruction])
        if segs:
            memory_parts.append(f"## Subtask Memory\n{seg_store.format_for_prompt(segs)}")

    if not memory_parts:
        return ""

    memory_block = "\n\n".join(memory_parts)
    if len(memory_block) > max_memory_chars:
        memory_block = memory_block[:max_memory_chars] + "\n... (truncated)"

    injection = (
        f"<memory>\n{memory_block}\n</memory>\n\n"
        "Use the memory above as reference for this turn. "
        "Adapt it to the current context rather than copying values literally."
    )

    task_id = test_entry.get("id", "?")
    print(
        f"\n{_C_BOLD}{_C_CYAN}{'━' * 80}{_C_RESET}\n"
        f"{_C_BOLD}{_C_CYAN}[DYNAMIC MEMORY] {task_id} | {turn_instruction[:60]}{_C_RESET}\n"
        f"{_C_CYAN}{injection}{_C_RESET}\n"
        f"{_C_BOLD}{_C_CYAN}{'━' * 80}{_C_RESET}",
        flush=True,
    )
    return injection


def _format_func_description(func_desc: dict) -> str:
    """Format a function description dict into a concise schema string."""
    name = func_desc.get("name", "")
    desc = func_desc.get("description", "")
    params = func_desc.get("parameters", {})
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


def augment_with_function_hints(
    execution_results: list[str],
    decoded_model_responses: list,
    memory_dir: Path,
    turn_query: str = "",
    func_descs: Optional[dict] = None,
) -> list[str]:
    """
    For each execution result that is an error, append a function memory hint
    and the function's schema description.
    Returns a new list with hints appended to error results.
    """
    stores = _get_stores(memory_dir)
    func_store = stores["func"]

    augmented = []
    n_errors = 0
    n_hints = 0
    for i, result in enumerate(execution_results):
        if not result.startswith("Error during execution:"):
            augmented.append(result)
            continue

        n_errors += 1
        hint = ""
        func_name = ""
        if i < len(decoded_model_responses):
            func_call_str = str(decoded_model_responses[i])
            func_name = _extract_func_name_from_call(func_call_str)
            if func_name:
                records = func_store.retrieve(func_name, turn_query=turn_query, max_examples=1)
                if records:
                    hint = func_store.format_hint(records, func_name)

        error_msg = result[len("Error during execution:"):].strip()
        hint_parts = [result, f"\n[Memory Hint] Your call to `{func_name}` failed with: {error_msg}"]

        # Function description from test_entry
        if func_descs and func_name and func_name in func_descs:
            desc_text = _format_func_description(func_descs[func_name])
            hint_parts.append(desc_text)

        if hint:
            n_hints += 1
            hint_parts.append(hint)

        if len(hint_parts) > 2:
            hint_block = "\n".join(hint_parts)
            print(
                f"\n{_C_BOLD}{_C_YELLOW}{'━' * 80}{_C_RESET}\n"
                f"{_C_BOLD}{_C_YELLOW}[FUNC MEMORY HINT] {func_name}{_C_RESET}\n"
                f"{_C_CYAN}{'  '.join(hint_parts[1:])}{_C_RESET}\n"
                f"{_C_BOLD}{_C_YELLOW}{'━' * 80}{_C_RESET}",
                flush=True,
            )
            augmented.append(hint_block)
        else:
            augmented.append(result)

    if n_errors > 0:
        print(f"[func_memory] errors={n_errors} hints_injected={n_hints}", flush=True, file=sys.stderr)

    return augmented
