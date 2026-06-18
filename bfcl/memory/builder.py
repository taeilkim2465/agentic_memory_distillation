"""
Build BFCL teacher memories from teacher inference result files.

Usage:
    builder = BFCLMemoryBuilder(
        memory_dir=Path("data/memory/0515"),
        teacher_llm="claude-sonnet-4-6",
        embedding_model="text-embedding-3-small",
    )
    builder.build_from_result_file(result_file, test_entries_by_id)
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from bfcl_eval.memory.data_model import (
    FunctionStepRecord,
    SubtaskSegment,
    WorkflowMemDoc,
    local_embedding,
    sanitize_memory_text,
)

_WORKFLOW_SYSTEM_PROMPT = (
    "You generate concise, reusable memory insights for future AI agents."
)

_WORKFLOW_PROMPT = """\
You are provided with a multi-turn function-calling task and its conversation trajectory.
The agent used various tools to complete the user's request.

Your job: generate a retrieval-friendly memory insight for future similar tasks.

Rules:
- Output exactly one short paragraph (2-4 sentences).
- Keep wording principle-level and reusable, not specific to this exact task.
- Mention key tool names only when broadly useful.
- Only include tools/actions that appear in the trajectory; do not invent steps.
- Focus on decision rules, the correct sequence of actions, and what to verify.
- Replace concrete values with placeholders (e.g., <FILE_PATH>, <USER_ID>).
- Include at least one validation cue and one common failure pattern to avoid.
- Keep it under 120 words.
- Output only the paragraph (no bullets, no markdown).

Task Query: {query}

Conversation Trajectory:
{trajectory}
"""

_SUBTASK_SYSTEM_PROMPT = (
    "You are analysing a successful function-calling trajectory to identify natural sub-task boundaries."
)

_SUBTASK_PROMPT = """\
Task query: {query}
Available tools: {tool_names}

Trajectory ({n_steps} tool-call steps):
{trajectory}

Identify the sub-task segments. Each segment should be a coherent, self-contained unit of work
(e.g. "navigate to directory", "create file", "search for content", "verify result").

Rules:
- A segment must contain at least one step.
- Steps must not overlap or be skipped; together they must cover all {n_steps} steps.
- Exploratory/verification steps (e.g. ls, pwd, grep) belong to whichever segment uses that information.
- Label: short verb-phrase (≤ 5 words), e.g. "list directory contents".
- Description: one sentence explaining what this segment accomplishes and why.

Reply with ONLY a JSON array — no prose, no markdown fences:
[
  {{"start_step": 1, "end_step": 3, "label": "navigate to directory", "description": "Change to the target directory and confirm the current location."}},
  {{"start_step": 4, "end_step": 7, "label": "create and write file", "description": "Create a new file and write the required content into it."}},
  ...
]
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_tool_name(call: Any) -> str:
    """Extract function name from a decoded tool call.

    Handles two formats produced by BFCL FC handlers:
      - dict:   {'function_name': 'args_json'}  → key is the name
      - string: 'function_name(arg=val, ...)'   → leading word is the name
    """
    if isinstance(call, dict):
        keys = [k for k in call if k not in ("name", "arguments", "id", "type")]
        if keys:
            return keys[0]
        # OpenAI-style {"name": "fn", "arguments": "..."}
        return str(call.get("name", ""))
    call_str = str(call)
    m = re.match(r"[\w.]+", call_str)
    return m.group(0).split(".")[-1] if m else ""


def _call_llm(
    llm: str,
    system: str,
    user: str,
    temperature: float = 0.0,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    import litellm

    litellm.drop_params = True
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    extra: dict = {}
    if api_base:
        extra["api_base"] = api_base
    if api_key:
        extra["api_key"] = api_key
    resp = litellm.completion(
        model=llm,
        messages=messages,
        temperature=temperature,
        timeout=60.0,
        **extra,
    )
    # Strip <think>...</think> blocks (Qwen3 style)
    content = resp.choices[0].message.content or ""
    content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL).strip()
    if "<think>" in content:
        content = content[: content.index("<think>")].strip()
    return content


def _embed(text: str, model: str = "text-embedding-3-small") -> list[float]:
    import litellm
    resp = litellm.embedding(model=model, input=[text])
    return [float(v) for v in resp.data[0]["embedding"]]


# ---------------------------------------------------------------------------
# Trajectory formatting
# ---------------------------------------------------------------------------

def format_trajectory_from_inference_log(
    test_entry: dict,
    result_entry: dict,
) -> tuple[str, list[dict]]:
    """
    Format a BFCL result entry into a readable trajectory string and flat step list.

    Returns (trajectory_text, steps) where each step is:
      {"turn": int, "step": int, "type": "user"|"tool_call"|"tool_result", "content": str, "tool_name": str}
    """
    lines: list[str] = []
    steps: list[dict] = []
    global_step = 0

    question_turns: list[list[dict]] = test_entry.get("question", [])
    inference_log: list[Any] = result_entry.get("inference_log", [])

    turn_idx = 0
    log_idx = 0

    while turn_idx < len(question_turns) and log_idx < len(inference_log):
        turn_msgs = question_turns[turn_idx]
        turn_log = inference_log[log_idx]

        # Skip state_info logs
        if isinstance(turn_log, list) and turn_log and isinstance(turn_log[0], dict) and turn_log[0].get("role") == "state_info":
            log_idx += 1
            continue

        # User message from question
        for msg in turn_msgs:
            if msg.get("role") == "user":
                global_step += 1
                content = msg.get("content", "")
                lines.append(f"[Turn {turn_idx+1} - User]\n{content}\n")
                steps.append({
                    "turn": turn_idx, "step": global_step,
                    "type": "user", "content": content, "tool_name": "",
                })

        # Steps from inference log
        if isinstance(turn_log, dict):
            # Turn-level think generated once before the step loop during teacher inference
            turn_think = turn_log.get("turn_think", "")

            step_keys = sorted(
                [k for k in turn_log if k.startswith("step_")],
                key=lambda k: int(k.split("_")[1]),
            )
            for sk in step_keys:
                step_entries = turn_log[sk]
                if not isinstance(step_entries, list):
                    continue
                for entry in step_entries:
                    role = entry.get("role", "")
                    if role == "assistant":
                        content = entry.get("content", "")
                        if content:
                            global_step += 1
                            lines.append(f"[Turn {turn_idx+1} - Agent]\n{content}\n")
                            steps.append({
                                "turn": turn_idx, "step": global_step,
                                "type": "tool_call", "content": str(content), "tool_name": "",
                                "think": turn_think,
                            })
                    elif role == "tool":
                        content = str(entry.get("content", ""))
                        global_step += 1
                        lines.append(f"[Turn {turn_idx+1} - Tool Result]\n{content}\n")
                        steps.append({
                            "turn": turn_idx, "step": global_step,
                            "type": "tool_result", "content": content, "tool_name": "",
                        })

        turn_idx += 1
        log_idx += 1

    # Try to attach tool names to tool_call steps from decoded responses
    result_turns: list[list] = result_entry.get("result", [])
    for t_idx, turn_results in enumerate(result_turns):
        for turn_step in turn_results:
            # turn_step is a list of decoded tool calls for this step
            if not isinstance(turn_step, list) or not turn_step:
                continue
            tool_name = _extract_tool_name(turn_step[0])
            if not tool_name:
                continue
            for s in steps:
                if s["type"] == "tool_call" and s["turn"] == t_idx and not s["tool_name"]:
                    s["tool_name"] = tool_name
                    break

    return "\n".join(lines), steps


# ---------------------------------------------------------------------------
# Memory builders
# ---------------------------------------------------------------------------

def build_workflow_memory(
    task_id: str,
    category: str,
    query: str,
    trajectory: str,
    involved_classes: list[str],
    llm: str,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[WorkflowMemDoc]:
    user_prompt = _WORKFLOW_PROMPT.format(query=query, trajectory=trajectory)
    try:
        insight = _call_llm(llm, _WORKFLOW_SYSTEM_PROMPT, user_prompt, api_base=api_base, api_key=api_key)
        insight = insight.strip()
    except Exception as e:
        print(f"[builder] workflow LLM failed for {task_id}: {e}", flush=True)
        return None

    if not insight:
        return None

    return WorkflowMemDoc(
        task_id=task_id,
        category=category,
        query=query,
        insight=insight,
        involved_classes=involved_classes,
        source="teacher",
        metadata={"generated_at": _utc_now()},
    )


_ARG_PLACEHOLDER_MAP = [
    (re.compile(r"email|mail",           re.I), "<EMAIL>"),
    (re.compile(r"password|passwd|pwd",  re.I), "<PASSWORD>"),
    (re.compile(r"dir|folder|directory", re.I), "<DIR>"),
    (re.compile(r"file|path|src|dest|source|destination|target", re.I), "<FILE_PATH>"),
    (re.compile(r"\bid\b|_id$",          re.I), "<ID>"),
    (re.compile(r"name|title|label",     re.I), "<NAME>"),
    (re.compile(r"phone|tel|number",     re.I), "<PHONE>"),
    (re.compile(r"key|keyword|query|pattern|search", re.I), "<QUERY>"),
    (re.compile(r"content|body|text|message", re.I), "<CONTENT>"),
    (re.compile(r"user|author|owner",    re.I), "<USER>"),
    (re.compile(r"date|time|timestamp",  re.I), "<DATETIME>"),
    (re.compile(r"url|link|href",        re.I), "<URL>"),
]

_EMAIL_RE   = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_FILE_EXT_RE = re.compile(r"\S+\.[a-zA-Z]{2,4}$")
_LONG_NUM_RE = re.compile(r"\b\d{5,}\b")


def _mask_arg_value(key: str, value: Any) -> Any:
    """Replace a single argument value with a typed placeholder."""
    if isinstance(value, bool) or not isinstance(value, str):
        return value  # booleans and numbers are not sensitive
    # Key-based matching
    for pattern, placeholder in _ARG_PLACEHOLDER_MAP:
        if pattern.search(key):
            return placeholder
    # Value-pattern fallback
    if _EMAIL_RE.search(value):
        return "<EMAIL>"
    if _FILE_EXT_RE.match(value.strip()):
        return "<FILE_PATH>"
    if _LONG_NUM_RE.match(value.strip()):
        return "<ID>"
    # Generic string → use uppercased key as placeholder type
    return f"<{key.upper()}>" if key else "<VALUE>"


def _extract_args_for_tool(tool_name: str, content: str) -> dict | None:
    """Extract the args dict for a specific tool from BFCL content.

    BFCL tool_call content is a stringified list of dicts:
      [{'tool_name': '{"arg": "val"}'}, ...]
    Returns:
      dict  — args for the matching tool (may be {} for no-arg tools like pwd)
      None  — tool not found or parse failed entirely
    """
    try:
        import ast
        calls = ast.literal_eval(content) if content.startswith("[") else json.loads(content)
        if isinstance(calls, list):
            for call in calls:
                if isinstance(call, dict) and tool_name in call:
                    val = call[tool_name]
                    return json.loads(val) if isinstance(val, str) else val
        elif isinstance(calls, dict):
            return calls
    except Exception:
        pass
    return None


def _mask_call_args(tool_name: str, content: str) -> str:
    """Parse args for the specific tool and replace values with typed placeholders."""
    args = _extract_args_for_tool(tool_name, content)
    if args is None:
        # Parse failed — show sanitised snippet rather than raw list
        return sanitize_memory_text(content[:100])
    if not args:
        return ""   # no-arg tool (e.g. pwd())
    masked = {k: _mask_arg_value(k, v) for k, v in args.items()}
    return json.dumps(masked, ensure_ascii=False)


def _format_numbered_trajectory(
    call_steps: list[dict],
    result_steps: list[dict],
    all_steps: Optional[list[dict]] = None,
) -> str:
    """Format tool-call steps as a numbered list, interleaving user messages per turn."""
    # Collect user message per turn
    user_msgs: dict[int, str] = {}
    for s in (all_steps or []):
        if s["type"] == "user" and s["turn"] not in user_msgs:
            user_msgs[s["turn"]] = s["content"]

    lines = []
    last_turn = -1
    for i, cs in enumerate(call_steps):
        turn = cs["turn"]
        if turn != last_turn:
            if turn in user_msgs:
                lines.append(f"\n[Turn {turn + 1} - User]: {user_msgs[turn]}")
            last_turn = turn
        tool = cs.get("tool_name", "?")
        call_content = cs.get("content", "")
        obs = result_steps[i]["content"][:300] if i < len(result_steps) else ""
        obs_preview = obs.replace("\n", " ") + ("…" if len(obs) == 300 else "")
        lines.append(
            f"Step {i + 1}: {tool}({call_content[:200]})\n"
            f"  → {obs_preview}"
        )
    return "\n".join(lines)


def build_subtask_segments(
    task_id: str,
    category: str,
    query: str,
    trajectory: str,
    tool_names: list[str],
    n_steps: int,
    llm: str,
    steps: Optional[list[dict]] = None,
    involved_classes: Optional[list[str]] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
) -> list[SubtaskSegment]:
    call_steps = [s for s in (steps or []) if s["type"] == "tool_call"]
    result_steps = [s for s in (steps or []) if s["type"] == "tool_result"]
    n_call_steps = len(call_steps)

    if n_call_steps == 0:
        return []

    numbered_traj = _format_numbered_trajectory(call_steps, result_steps, all_steps=steps)
    user_prompt = _SUBTASK_PROMPT.format(
        query=query,
        tool_names=", ".join(tool_names),
        n_steps=n_call_steps,
        trajectory=numbered_traj,
    )
    try:
        raw = _call_llm(llm, _SUBTASK_SYSTEM_PROMPT, user_prompt, temperature=0.2, api_base=api_base, api_key=api_key)
    except Exception as e:
        print(f"[builder] subtask LLM failed for {task_id}: {e}", flush=True)
        return []

    raw = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"^```(?:json)?\n?", "", raw).rstrip("`").strip()
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return []

    try:
        seg_list = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []

    segments: list[SubtaskSegment] = []
    now = _utc_now()
    for seg in seg_list:
        if not isinstance(seg, dict) or not seg.get("label"):
            continue

        start = max(0, int(seg.get("start_step", 1)) - 1)
        end = max(start + 1, min(int(seg.get("end_step", n_call_steps)), n_call_steps))
        seg_calls = call_steps[start:end]
        seg_results = result_steps[start:end]

        # Masked tool call strings: "tool_name({masked_args})"
        masked_tool_calls = []
        for cs in seg_calls:
            tool = cs.get("tool_name", "")
            masked_args = _mask_call_args(tool, cs.get("content", ""))
            masked_tool_calls.append(f"{tool}({masked_args})" if tool else masked_args)

        # Sanitize observations (email, phone, long IDs)
        masked_observations = [
            sanitize_memory_text(rs["content"][:500]) for rs in seg_results
        ]

        required_tools = list(dict.fromkeys(
            cs["tool_name"] for cs in seg_calls if cs.get("tool_name")
        ))

        excerpt_lines = [
            f"→ {cs.get('tool_name', '?')}: {sanitize_memory_text(rs['content'][:200])}"
            for cs, rs in zip(seg_calls, seg_results)
        ]

        segments.append(
            SubtaskSegment(
                task_id=task_id,
                category=category,
                label=str(seg.get("label", "")).strip(),
                description=str(seg.get("description", "")).strip(),
                tool_calls=masked_tool_calls,
                observations=masked_observations,
                required_tools=required_tools,
                trajectory_excerpt="\n".join(excerpt_lines),
                source="teacher",
                source_task_id=task_id,
                generated_at=now,
            )
        )
    return segments


_FUNC_ERROR_RE = re.compile(r"^Error during execution:", re.IGNORECASE)


def _is_error_result(result_str: str) -> bool:
    s = result_str.strip()
    if _FUNC_ERROR_RE.match(s):
        return True
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and "error" in obj:
            return True
    except Exception:
        pass
    return False


def build_function_records(
    task_id: str,
    category: str,
    query: str,
    steps: list[dict],
) -> list[FunctionStepRecord]:
    records: list[FunctionStepRecord] = []
    now = _utc_now()

    # Build turn_idx → user instruction map
    turn_instructions: dict[int, str] = {}
    for s in steps:
        if s["type"] == "user" and s["turn"] not in turn_instructions:
            turn_instructions[s["turn"]] = s["content"]

    call_steps = [s for s in steps if s["type"] == "tool_call"]
    result_steps = [s for s in steps if s["type"] == "tool_result"]

    for i, call_step in enumerate(call_steps):
        tool_name = call_step.get("tool_name", "")
        if not tool_name:
            continue

        result_str = result_steps[i]["content"] if i < len(result_steps) else ""
        result_str = result_str[:500]

        if _is_error_result(result_str):
            continue

        call_content = str(call_step.get("content", ""))
        raw_args = _extract_args_for_tool(tool_name, call_content) or {}
        args = {k: _mask_arg_value(k, v) for k, v in raw_args.items()}

        turn_instruction = turn_instructions.get(call_step["turn"], query)

        records.append(
            FunctionStepRecord(
                tool_name=tool_name,
                category=category,
                task_id=task_id,
                step=call_step["step"],
                arguments=args,
                result=sanitize_memory_text(result_str),
                task_query=query,
                turn_instruction=turn_instruction,
                think=call_step.get("think", ""),
                generated_at=now,
            )
        )
    return records




# ---------------------------------------------------------------------------
# Main builder class
# ---------------------------------------------------------------------------

class BFCLMemoryBuilder:
    """Build workflow, subtask, and function memories from teacher result files."""

    def __init__(
        self,
        memory_dir: Path,
        teacher_llm: str = "claude-sonnet-4-6",
        embedding_model: str = "text-embedding-3-small",
        teacher_llm_base_url: Optional[str] = None,
        teacher_llm_api_key: Optional[str] = None,
    ):
        self.memory_dir = Path(memory_dir)
        self.teacher_llm = teacher_llm
        self.embedding_model = embedding_model
        self.teacher_llm_base_url = teacher_llm_base_url
        self.teacher_llm_api_key = teacher_llm_api_key

        (self.memory_dir / "workflow").mkdir(parents=True, exist_ok=True)
        (self.memory_dir / "subtask").mkdir(parents=True, exist_ok=True)
        (self.memory_dir / "function").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def build_from_result_file(
        self,
        result_file: Path,
        test_entries_by_id: dict[str, dict],
        only_ok_status: bool = False,
        build_types: Optional[set[str]] = None,
        allowed_ids: Optional[set[str]] = None,
    ) -> dict[str, int]:
        """
        Process one teacher result JSON file and append to memory stores.

        build_types: subset of {"workflow", "subtask", "function"} to build.
                     None means build all.
        allowed_ids: if set, only process tasks whose IDs are in this set.
        Returns counts: {"workflow": N, "subtask": N, "function": N}
        """
        if build_types is None:
            build_types = {"workflow", "subtask", "function"}
        result_file = Path(result_file)
        results: list[dict] = []
        with result_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))

        counts = {"workflow": 0, "subtask": 0, "function": 0}
        workflow_docs: list[WorkflowMemDoc] = []
        subtask_segs: list[SubtaskSegment] = []
        func_records: list[FunctionStepRecord] = []

        for res in results:
            task_id = res.get("id", "")
            # decode_error는 마지막 턴에 텍스트 응답을 돌려준 것으로, 실제 tool call은 정상 실행됨
            if only_ok_status and res.get("generation_status") == "force_quit":
                continue
            if allowed_ids is not None and task_id not in allowed_ids:
                continue
            if task_id not in test_entries_by_id:
                continue

            test_entry = test_entries_by_id[task_id]
            category = task_id.rsplit("_", 1)[0]
            involved_classes = test_entry.get("involved_classes", [])
            tool_names = [f.get("name", "") for f in test_entry.get("function", []) if f.get("name")]

            trajectory, steps = format_trajectory_from_inference_log(test_entry, res)
            query = ""
            for turn in test_entry.get("question", []):
                for msg in turn:
                    if msg.get("role") == "user":
                        query = msg.get("content", "")
                        break
                if query:
                    break

            if not trajectory.strip():
                continue

            print(f"  [builder] {task_id} — {len(steps)} steps", flush=True)

            if "workflow" in build_types:
                wf = build_workflow_memory(
                    task_id=task_id,
                    category=category,
                    query=query,
                    trajectory=trajectory,
                    involved_classes=involved_classes,
                    llm=self.teacher_llm,
                    api_base=self.teacher_llm_base_url,
                    api_key=self.teacher_llm_api_key,
                )
                if wf:
                    workflow_docs.append(wf)
                    counts["workflow"] += 1

            if "subtask" in build_types:
                segs = build_subtask_segments(
                    task_id=task_id,
                    category=category,
                    query=query,
                    trajectory=trajectory,
                    tool_names=tool_names,
                    n_steps=len(steps),
                    llm=self.teacher_llm,
                    steps=steps,
                    api_base=self.teacher_llm_base_url,
                    api_key=self.teacher_llm_api_key,
                )
                subtask_segs.extend(segs)
                counts["subtask"] += len(segs)

            if "function" in build_types:
                recs = build_function_records(task_id, category, query, steps)
                func_records.extend(recs)
                counts["function"] += len(recs)

        if "workflow" in build_types:
            self._save_workflow(workflow_docs)
        if "subtask" in build_types:
            self._save_subtask(subtask_segs)
        if "function" in build_types:
            self._save_function(func_records)
        return counts

    # ------------------------------------------------------------------
    def _save_workflow(self, docs: list[WorkflowMemDoc]) -> None:
        if not docs:
            return
        out_path = self.memory_dir / "workflow" / "documents.json"
        emb_path = self.memory_dir / "workflow" / "embeddings.npy"

        # Merge with existing
        existing: list[dict] = []
        if out_path.exists():
            with out_path.open(encoding="utf-8") as f:
                existing = json.load(f)

        existing_ids = {d["task_id"] for d in existing}
        new_docs = [d for d in docs if d.task_id not in existing_ids]
        all_docs = existing + [d.to_dict() for d in new_docs]

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(all_docs, f, ensure_ascii=False, indent=2)

        # Rebuild full embedding array
        print(f"  [builder] embedding {len(all_docs)} workflow docs...")
        embeddings = [_embed(d["query"], self.embedding_model) for d in all_docs]
        np.save(str(emb_path), np.array(embeddings, dtype=np.float32))
        print(f"  [builder] saved {len(all_docs)} workflow docs → {out_path}")

    def _save_subtask(self, segs: list[SubtaskSegment]) -> None:
        if not segs:
            return
        seg_path = self.memory_dir / "subtask" / "segments.jsonl"
        emb_path = self.memory_dir / "subtask" / "segments_embeddings.npy"

        existing_segs: list[dict] = []
        if seg_path.exists():
            for line in seg_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        existing_segs.append(json.loads(line))
                    except Exception:
                        pass

        existing_ids = {(d["task_id"], d["label"]) for d in existing_segs}
        new_segs = [s for s in segs if (s.task_id, s.label) not in existing_ids]
        all_segs = existing_segs + [s.to_dict() for s in new_segs]

        with seg_path.open("w", encoding="utf-8") as f:
            for d in all_segs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

        print(f"  [builder] embedding {len(all_segs)} subtask segments...")
        embeddings = [
            _embed(f"{d.get('label', '')} {d.get('description', '')}".strip(), self.embedding_model)
            for d in all_segs
        ]
        np.save(str(emb_path), np.array(embeddings, dtype=np.float32))
        print(f"  [builder] saved {len(all_segs)} segments → {seg_path}")

    def _save_function(self, records: list[FunctionStepRecord]) -> None:
        if not records:
            return
        rec_path = self.memory_dir / "function" / "records.jsonl"
        emb_path = self.memory_dir / "function" / "records_embeddings.npy"

        existing: list[dict] = []
        if rec_path.exists():
            for line in rec_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        existing.append(json.loads(line))
                    except Exception:
                        pass

        all_records = existing + [r.to_dict() for r in records]

        with rec_path.open("w", encoding="utf-8") as f:
            for d in all_records:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

        print(f"  [builder] embedding {len(all_records)} function records...")
        embeddings = [
            _embed(d.get("turn_instruction", "") or d.get("task_query", ""), self.embedding_model)
            for d in all_records
        ]
        np.save(str(emb_path), np.array(embeddings, dtype=np.float32))
        print(f"  [builder] saved {len(all_records)} function records → {rec_path}")

