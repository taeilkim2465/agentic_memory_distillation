#!/usr/bin/env python3
"""
Generate Mem^p memories from an ACE teacher run.

Reads each task folder in the teacher output directory, reconstructs
the execution trajectory from lm_calls.jsonl, and calls gpt-5-mini to
produce keywords and a procedural script, then stores entries in the
memp memory store.

Usage:
    python generate_memp_memories.py \
        --tasks_dir /path/to/teacher/tasks \
        --store_path experiments/memory/memp_teacher_store.json \
        --model gpt-5-mini \
        [--workers 4] \
        [--skip_existing]
"""

import argparse
import fcntl
import json
import os
import re
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Inline MemoryEntry / MemoryStore (avoids complex import chain)
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    id: str
    task_id: str
    task_desc: str
    keywords: list
    trajectory: list
    script: str
    success: bool
    created_at: str
    updated_at: str

    @classmethod
    def create(cls, task_id, task_desc, keywords, trajectory, script, success):
        now = datetime.utcnow().isoformat()
        return cls(
            id=str(uuid.uuid4())[:8],
            task_id=task_id,
            task_desc=task_desc,
            keywords=keywords,
            trajectory=trajectory,
            script=script,
            success=success,
            created_at=now,
            updated_at=now,
        )


class MemoryStore:
    def __init__(self, store_path: str):
        self.store_path = store_path
        self.lock_path = store_path + ".lock"
        os.makedirs(os.path.dirname(os.path.abspath(store_path)), exist_ok=True)

    @contextmanager
    def _file_lock(self):
        with open(self.lock_path, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

    def _load(self):
        if os.path.exists(self.store_path):
            with open(self.store_path) as f:
                return json.load(f)
        return []

    def _save(self, entries):
        with open(self.store_path, "w") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)

    def existing_task_ids(self):
        with self._file_lock():
            return {e["task_id"] for e in self._load()}

    def add(self, entry: MemoryEntry):
        with self._file_lock():
            entries = self._load()
            entries.append(asdict(entry))
            self._save(entries)


# ---------------------------------------------------------------------------
# LLM helper (OpenAI client — supports gpt-5-mini reasoning models)
# ---------------------------------------------------------------------------

def llm_call(model: str, messages: list, max_completion_tokens: int = 4096) -> str:
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        temperature=1,
    )
    return (resp.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# Task parsing helpers
# ---------------------------------------------------------------------------

def parse_task_desc(task_dir: str) -> str:
    transcript = Path(task_dir) / "logs" / "conversation_transcript.md"
    text = transcript.read_text()
    m = re.search(r"- Instruction: (.+)", text)
    if m:
        return m.group(1).strip()
    raise ValueError(f"Cannot find Instruction in {transcript}")


def parse_success(task_dir: str) -> bool:
    report = Path(task_dir) / "evaluation" / "report.md"
    text = report.read_text()
    m = re.search(r"Num Failed Tests\s*:\s*(\d+)", text)
    if m:
        return int(m.group(1)) == 0
    raise ValueError(f"Cannot parse report in {report}")


def reconstruct_trajectory(task_dir: str) -> list:
    """
    Reconstruct the execution trajectory from lm_calls.jsonl.

    Each lm_call's input contains the cumulative messages up to that point.
    The last call is the proceduralize call (1-message input); we skip it.
    The remaining calls are execution calls. We take the last execution call's
    full input messages and append its output as the final assistant message.
    """
    lm_calls_path = Path(task_dir) / "logs" / "lm_calls.jsonl"
    calls = [json.loads(line) for line in lm_calls_path.read_text().splitlines() if line.strip()]

    # last call is the proceduralize call — skip it
    exec_calls = calls[:-1]
    if not exec_calls:
        return []

    last_exec = exec_calls[-1]
    trajectory = list(last_exec["input"]["messages"])

    # append the final assistant response
    out = last_exec.get("output", {})
    if isinstance(out, dict) and "choices" in out:
        content = out["choices"][0]["message"]["content"]
        trajectory.append({"role": "assistant", "content": content})

    return trajectory


# ---------------------------------------------------------------------------
# LLM-based keyword extraction and proceduralization
# ---------------------------------------------------------------------------

def load_prompt(path: str) -> str:
    return Path(path).read_text()


def extract_keywords(task_desc: str, prompt_template: str, model: str) -> list:
    prompt = prompt_template.replace("{{task}}", task_desc)
    raw = llm_call(model, [{"role": "user", "content": prompt}], max_completion_tokens=512)
    # strip json fences if present
    raw = re.sub(r"^```json\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed.get("keywords"), list):
            return [str(k) for k in parsed["keywords"]]
    except json.JSONDecodeError:
        pass
    # fallback: simple split
    words = re.findall(r"\b[a-z_]{3,}\b", task_desc.lower())
    return list(dict.fromkeys(words))[:8]


def proceduralize(task_desc: str, trajectory: list, success: bool,
                  prompt_template: str, model: str) -> str:
    conv_text = "\n".join(
        f"[{m['role'].upper()}]: {m['content']}" for m in trajectory
    )
    prompt = (
        prompt_template
        .replace("{{task}}", task_desc)
        .replace("{{trajectory}}", conv_text[:8000])
        .replace("{{success}}", "SUCCESS" if success else "FAILURE")
    )
    return llm_call(model, [{"role": "user", "content": prompt}], max_completion_tokens=2048)


# ---------------------------------------------------------------------------
# Per-task processor
# ---------------------------------------------------------------------------

def process_task(task_id: str, task_dir: str, store: MemoryStore,
                 keyword_prompt: str, proc_prompt: str, model: str,
                 skip_existing: bool) -> str:
    if skip_existing and task_id in store.existing_task_ids():
        return f"SKIP  {task_id} (already in store)"

    try:
        task_desc = parse_task_desc(task_dir)
        success = parse_success(task_dir)
        trajectory = reconstruct_trajectory(task_dir)

        keywords = extract_keywords(task_desc, keyword_prompt, model)
        script = proceduralize(task_desc, trajectory, success, proc_prompt, model)

        entry = MemoryEntry.create(
            task_id=task_id,
            task_desc=task_desc,
            keywords=keywords,
            trajectory=trajectory,
            script=script,
            success=success,
        )
        store.add(entry)
        outcome = "OK_S" if success else "OK_F"
        return f"{outcome}  {task_id} | {task_desc[:60]}"
    except Exception as e:
        return f"ERR   {task_id} | {e}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks_dir", required=True,
                        help="Path to teacher output tasks/ directory")
    parser.add_argument("--store_path",
                        default="experiments/memory/memp_teacher_store.json")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--keyword_prompt",
                        default="experiments/prompts/memp_keyword_prompt.txt")
    parser.add_argument("--proceduralize_prompt",
                        default="experiments/prompts/memp_proceduralize_prompt.txt")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip tasks already present in the store")
    args = parser.parse_args()

    tasks_dir = Path(args.tasks_dir)
    task_entries = sorted(
        [(d.name, str(d)) for d in tasks_dir.iterdir() if d.is_dir()]
    )
    print(f"Found {len(task_entries)} tasks in {tasks_dir}")

    store = MemoryStore(args.store_path)
    keyword_prompt = load_prompt(args.keyword_prompt)
    proc_prompt = load_prompt(args.proceduralize_prompt)

    def run(item):
        task_id, task_dir = item
        return process_task(task_id, task_dir, store,
                            keyword_prompt, proc_prompt, args.model,
                            args.skip_existing)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, item): item[0] for item in task_entries}
        for i, fut in enumerate(as_completed(futures), 1):
            result = fut.result()
            print(f"[{i:3d}/{len(task_entries)}] {result}")

    entries = json.load(open(args.store_path))
    print(f"\nDone. Store now has {len(entries)} entries → {args.store_path}")


if __name__ == "__main__":
    main()
