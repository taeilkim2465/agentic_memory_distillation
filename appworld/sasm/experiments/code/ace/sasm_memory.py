"""
SASM Memory Bank for AppWorld Teacher-Student Setup.

Stores (z, d, e) triples where:
  z = subtask category (explore/authenticate/query/execute/verify)
  d = subtask description (objective + keywords)
  e = abstracted transferable experience
"""

import json
import os
from dataclasses import dataclass
from typing import Optional

CATEGORIES = ["explore", "authenticate", "query", "execute", "verify"]


@dataclass
class MemoryEntry:
    z: str
    d: str
    e: str


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def _cosine_similarity_np(a: list[float], b: list[float]) -> float:
    import numpy as np
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


class SASMMemoryBank:
    """Subtask-level memory bank with category-gated retrieval."""

    def __init__(self, memory_file_path: str):
        self.memory_file_path = memory_file_path
        self.entries: list[MemoryEntry] = []
        self._embeddings: list[list[float]] = []
        self._encoder = None  # lazy-loaded
        if os.path.exists(memory_file_path):
            self.load()

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
                print("[SASM] Using sentence-transformers encoder.")
            except ImportError:
                self._encoder = "jaccard"
                print("[SASM] sentence-transformers not found; using Jaccard similarity.")
        return self._encoder

    def _encode(self, text: str) -> list[float]:
        encoder = self._get_encoder()
        if encoder == "jaccard":
            return []  # no pre-computed vector needed for Jaccard
        return encoder.encode(text).tolist()

    def _similarity(self, query_text: str, candidate_text: str,
                    query_emb: list[float], candidate_emb: list[float]) -> float:
        if self._encoder == "jaccard" or not query_emb or not candidate_emb:
            return _jaccard_similarity(query_text, candidate_text)
        return _cosine_similarity_np(query_emb, candidate_emb)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def add(self, z: str, d: str, e: str) -> None:
        if z not in CATEGORIES:
            print(f"[SASM] Warning: unknown category '{z}'. Using 'explore'.")
            z = "explore"
        entry = MemoryEntry(z=z, d=d, e=e)
        self.entries.append(entry)
        self._embeddings.append(self._encode(d))
        self.save()

    def retrieve(self, z: str, d_query: str) -> Optional[str]:
        """Category-gated top-1 retrieval by semantic similarity."""
        query_emb = self._encode(d_query)
        filtered = [
            (i, entry) for i, entry in enumerate(self.entries) if entry.z == z
        ]
        if not filtered:
            return None

        best_score = -1.0
        best_entry = None
        for i, entry in filtered:
            score = self._similarity(d_query, entry.d, query_emb, self._embeddings[i])
            if score > best_score:
                best_score = score
                best_entry = entry

        return best_entry.e if best_entry else None

    def stats(self) -> dict:
        counts: dict[str, int] = {z: 0 for z in CATEGORIES}
        for entry in self.entries:
            counts[entry.z] = counts.get(entry.z, 0) + 1
        return {"total": len(self.entries), "by_category": counts}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.memory_file_path)), exist_ok=True)
        data = {
            "entries": [{"z": e.z, "d": e.d, "e": e.e} for e in self.entries],
            "embeddings": self._embeddings,
        }
        with open(self.memory_file_path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self) -> None:
        with open(self.memory_file_path) as f:
            data = json.load(f)
        self.entries = [MemoryEntry(**item) for item in data.get("entries", [])]
        self._embeddings = data.get("embeddings", [])
        while len(self._embeddings) < len(self.entries):
            idx = len(self._embeddings)
            self._embeddings.append(self._encode(self.entries[idx].d))
        print(f"[SASM] Loaded {len(self.entries)} memory entries from {self.memory_file_path}")
