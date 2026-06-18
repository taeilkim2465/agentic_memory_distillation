"""
Two-layer memory store for Mem^p.

Each MemoryEntry holds:
  - trajectory : concrete step-by-step execution history (list of messages)
  - script     : abstract procedural summary derived from the trajectory

The store is persisted as a plain JSON file so any model can read it
(model-agnostic, Concept 5 of the Mem^p paper).
"""

import fcntl
import json
import os
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class MemoryEntry:
    id: str
    task_id: str
    task_desc: str
    keywords: list
    trajectory: list          # list of {role, content} messages
    script: str               # abstract procedural summary
    success: bool
    created_at: str
    updated_at: str

    @classmethod
    def create(
        cls,
        task_id: str,
        task_desc: str,
        keywords: list,
        trajectory: list,
        script: str,
        success: bool,
    ) -> "MemoryEntry":
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
    """JSON-backed memory store supporting add and in-place update (Concept 3)."""

    def __init__(self, store_path: str):
        self.store_path = store_path
        self.lock_path = store_path + ".lock"
        self.entries: list = []
        self._load()

    @contextmanager
    def _file_lock(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.lock_path)), exist_ok=True)
        with open(self.lock_path, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

    def _load(self):
        if os.path.exists(self.store_path):
            with open(self.store_path) as f:
                raw = json.load(f)
            self.entries = [MemoryEntry(**e) for e in raw]
        else:
            self.entries = []

    def _load_from_disk(self) -> list:
        if os.path.exists(self.store_path):
            with open(self.store_path) as f:
                raw = json.load(f)
            return [MemoryEntry(**e) for e in raw]
        return []

    def _save_entries(self, entries: list):
        os.makedirs(os.path.dirname(os.path.abspath(self.store_path)), exist_ok=True)
        with open(self.store_path, "w") as f:
            json.dump([asdict(e) for e in entries], f, indent=2, ensure_ascii=False)

    def add(self, entry: MemoryEntry) -> None:
        with self._file_lock():
            entries = self._load_from_disk()
            entries.append(entry)
            self._save_entries(entries)
        self.entries.append(entry)

    def update_script(self, entry_id: str, new_script: str) -> bool:
        """In-place script update — the Adjustment strategy (Concept 3)."""
        with self._file_lock():
            entries = self._load_from_disk()
            for entry in entries:
                if entry.id == entry_id:
                    entry.script = new_script
                    entry.updated_at = datetime.utcnow().isoformat()
                    self._save_entries(entries)
                    # sync local cache
                    for local in self.entries:
                        if local.id == entry_id:
                            local.script = new_script
                            local.updated_at = entry.updated_at
                    return True
        return False

    def get_all(self) -> list:
        return list(self.entries)

    def __len__(self) -> int:
        return len(self.entries)
