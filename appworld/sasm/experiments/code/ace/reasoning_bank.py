"""
Reasoning Bank: stores reasoning experiences from successful and failed tasks.
Memory entry structure follows the paper: {title, description, content} per item,
up to 3 items per trajectory.
Retrieval uses OpenAI embeddings + cosine similarity on task query.
"""
import json
import os
import re
from datetime import datetime

import numpy as np
import openai


EMBEDDING_MODEL = "text-embedding-3-small"


def _get_embedding(text: str) -> list[float]:
    api_key = os.environ.get("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key, base_url="https://api.openai.com/v1")
    response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return response.data[0].embedding


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def parse_memory_items(markdown_text: str) -> list[dict]:
    """Parse the paper's Markdown output format into a list of {title, description, content}."""
    items = []
    # Split on '# Memory Item N'
    blocks = re.split(r"#\s*Memory Item\s+\d+", markdown_text, flags=re.IGNORECASE)
    for block in blocks:
        if not block.strip():
            continue
        title = re.search(r"##\s*Title\s+(.*?)(?=##|$)", block, re.DOTALL | re.IGNORECASE)
        description = re.search(r"##\s*Description\s+(.*?)(?=##|$)", block, re.DOTALL | re.IGNORECASE)
        content = re.search(r"##\s*Content\s+(.*?)(?=##|$)", block, re.DOTALL | re.IGNORECASE)
        if title or description or content:
            items.append({
                "title": title.group(1).strip() if title else "",
                "description": description.group(1).strip() if description else "",
                "content": content.group(1).strip() if content else "",
            })
    return items


def load_bank(file_path: str) -> list[dict]:
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return []


def save_bank(bank: list[dict], file_path: str) -> None:
    with open(file_path, "w") as f:
        json.dump(bank, f, indent=2)


def append_to_bank_safe(
    file_path: str,
    task_id: str,
    task_description: str,
    outcome: str,
    memory_items: list[dict],
) -> list[dict]:
    """Append a single entry to the bank file with exclusive file lock (race-safe for parallel processes)."""
    import fcntl

    embedding = None
    try:
        embedding = _get_embedding(task_description)
    except Exception as e:
        print(f"[ReasoningBank] Warning: failed to compute embedding: {e}")

    entry = {
        "task_id": task_id,
        "task_description": task_description,
        "outcome": outcome,
        "memory_items": memory_items[:3],
        "embedding": embedding,
        "timestamp": datetime.now().isoformat(),
    }

    lock_path = file_path + ".lock"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            bank = load_bank(file_path)
            bank.append(entry)
            save_bank(bank, file_path)
            print(f"[ReasoningBank] Saved {len(memory_items)} items (total entries: {len(bank)}) → {file_path}")
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)

    return bank


def add_memory(
    bank: list[dict],
    task_id: str,
    task_description: str,
    outcome: str,
    memory_items: list[dict],
) -> list[dict]:
    """
    Add a new entry to the bank.
    memory_items: list of {title, description, content} dicts (max 3, per paper).
    """
    if not memory_items:
        return bank

    embedding = None
    try:
        embedding = _get_embedding(task_description)
    except Exception as e:
        print(f"[ReasoningBank] Warning: failed to compute embedding: {e}")

    entry = {
        "task_id": task_id,
        "task_description": task_description,
        "outcome": outcome,  # "success" or "failure"
        "memory_items": memory_items[:3],  # at most 3 per paper
        "embedding": embedding,
        "timestamp": datetime.now().isoformat(),
    }
    bank.append(entry)
    return bank


def retrieve_memories(
    bank: list[dict],
    query: str,
    top_k: int = 3,
) -> list[dict]:
    """Return top_k bank entries most similar to query.
    Falls back to most recent top_k entries if embeddings unavailable.
    Each returned entry contains memory_items (list of {title, description, content}).
    """
    if not bank:
        return []

    entries_with_embedding = [e for e in bank if e.get("embedding")]

    if not entries_with_embedding:
        return bank[-top_k:]

    try:
        query_embedding = _get_embedding(query)
    except Exception as e:
        print(f"[ReasoningBank] Warning: failed to embed query, falling back to recency: {e}")
        return bank[-top_k:]

    scored = [
        (_cosine_similarity(query_embedding, entry["embedding"]), entry)
        for entry in entries_with_embedding
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [entry for _, entry in scored[:top_k]]


def format_memories_for_prompt(entries: list[dict]) -> str:
    """Format retrieved bank entries into prompt text.
    Flattens all memory_items across entries into a numbered list.
    """
    if not entries:
        return "(No past experiences yet)"

    parts = []
    item_num = 1
    for entry in entries:
        for item in entry.get("memory_items", []):
            parts.append(
                f"# Memory Item {item_num}\n"
                f"## Title {item.get('title', '')}\n"
                f"## Description {item.get('description', '')}\n"
                f"## Content {item.get('content', '')}"
            )
            item_num += 1

    return "\n\n".join(parts) if parts else "(No past experiences yet)"


def extract_json_from_text(text: str) -> dict | None:
    if text is None:
        return None
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    json_pattern = r"```json\s*(.*?)\s*```"
    matches = re.findall(json_pattern, text, re.DOTALL | re.IGNORECASE)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    # balanced brace extraction
    i = 0
    while i < len(text):
        if text[i] == "{":
            brace_count = 1
            start = i
            i += 1
            while i < len(text) and brace_count > 0:
                if text[i] == "{":
                    brace_count += 1
                elif text[i] == "}":
                    brace_count -= 1
                elif text[i] == '"':
                    i += 1
                    while i < len(text) and text[i] != '"':
                        if text[i] == "\\":
                            i += 1
                        i += 1
                i += 1
            if brace_count == 0:
                try:
                    return json.loads(text[start:i])
                except json.JSONDecodeError:
                    pass
        else:
            i += 1
    return None
