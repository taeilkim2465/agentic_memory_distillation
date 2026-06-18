"""
AveFact retrieval for Mem^p (Concept 2).

Algorithm:
  1. Keywords are extracted from the incoming task description (by the agent).
  2. Each keyword is embedded with a neural embedding model (e.g. OpenAI).
  3. Each stored entry is embedded (task_desc + keywords); embeddings are cached on disk.
  4. For each entry: average cosine similarity across all keyword embeddings → AveFact score.
  5. Return top-k entries sorted by score.

Falls back to TF-weighted token-overlap cosine when no embedding model is configured.
"""

import json
import os
import re
from collections import Counter
from math import sqrt
from typing import Optional

from .memory_store import MemoryEntry, MemoryStore

_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "including", "until", "against", "among",
    "and", "but", "or", "not", "my", "your", "his", "her", "its", "our",
    "their", "that", "this", "these", "those", "i", "you", "he", "she",
    "it", "we", "they", "me", "him", "us", "them", "what", "which", "who",
}


def _tokenize(text: str) -> list:
    return re.findall(r"\w+", text.lower())


def _cosine_tf(query_tokens: list, doc_tokens: list) -> float:
    """TF-weighted token-overlap cosine (fallback, no embedding model needed)."""
    if not query_tokens or not doc_tokens:
        return 0.0
    qf = Counter(query_tokens)
    df = Counter(doc_tokens)
    dot = sum(qf[t] * df[t] for t in qf if t in df)
    q_norm = sqrt(sum(v * v for v in qf.values()))
    d_norm = sqrt(sum(v * v for v in df.values()))
    if q_norm == 0 or d_norm == 0:
        return 0.0
    return dot / (q_norm * d_norm)


def _cosine_vec(a: list, b: list) -> float:
    """Cosine similarity between two float vectors."""
    import numpy as np
    va, vb = np.array(a, dtype=float), np.array(b, dtype=float)
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def extract_keywords_simple(text: str) -> list:
    """Heuristic keyword extraction (fallback when LLM is unavailable)."""
    words = re.findall(r"\b[A-Za-z][a-z]{2,}\b", text)
    return list({w.lower() for w in words if w.lower() not in _STOP_WORDS})


class AveFact:
    """
    Retrieve memories by averaging per-keyword embedding similarities (paper algorithm).

    Per the paper: extract key facts/keywords from the query, embed each one,
    then score each memory entry by the average cosine similarity across keywords.
    This captures which parts of a memory are relevant better than a single
    holistic embedding.

    Entry embedding text: entry.task_desc + " " + " ".join(entry.keywords)
    Entry embeddings are cached in a sidecar file (<store_path>.embeddings.json).
    """

    def __init__(
        self,
        store: MemoryStore,
        top_k: int = 3,
        embedding_model: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
    ):
        self.store = store
        self.top_k = top_k
        self.embedding_model = embedding_model
        self._embedding_api_key = embedding_api_key or os.environ.get("OPENAI_API_KEY")
        self._embedding_base_url = embedding_base_url
        self._client = None
        # entry.id → embedding vector (persisted as sidecar JSON)
        self._emb_cache: dict[str, list[float]] = {}
        self._cache_path = store.store_path + ".embeddings.json"
        if self.embedding_model:
            self._load_emb_cache()

    # ------------------------------------------------------------------
    # OpenAI embedding client
    # ------------------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            import openai
            kwargs: dict = {}
            if self._embedding_api_key:
                kwargs["api_key"] = self._embedding_api_key
            if self._embedding_base_url:
                kwargs["base_url"] = self._embedding_base_url
            self._client = openai.OpenAI(**kwargs)
        return self._client

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = self._get_client().embeddings.create(
            model=self.embedding_model, input=texts
        )
        return [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]

    # ------------------------------------------------------------------
    # Embedding cache (sidecar JSON keyed by entry.id)
    # ------------------------------------------------------------------

    def _load_emb_cache(self) -> None:
        if os.path.exists(self._cache_path):
            with open(self._cache_path) as f:
                self._emb_cache = json.load(f)

    def _save_emb_cache(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self._cache_path)), exist_ok=True)
        with open(self._cache_path, "w") as f:
            json.dump(self._emb_cache, f)

    @staticmethod
    def _entry_text(entry: MemoryEntry) -> str:
        kw_str = " ".join(entry.keywords)
        return f"{entry.task_desc} {kw_str}".strip()

    def _ensure_entry_embeddings(self, entries: list[MemoryEntry]) -> None:
        """Compute and cache embeddings for any entries missing from the cache."""
        missing = [e for e in entries if e.id not in self._emb_cache]
        if not missing:
            return
        texts = [self._entry_text(e) for e in missing]
        print(f"[AveFact] Embedding {len(missing)} new entries with {self.embedding_model}...")
        vectors = self._embed_batch(texts)
        for entry, vec in zip(missing, vectors):
            self._emb_cache[entry.id] = vec
        self._save_emb_cache()

    # ------------------------------------------------------------------
    # Core retrieval
    # ------------------------------------------------------------------

    def retrieve(self, task_desc: str, keywords: list) -> list:
        entries = self.store.get_all()
        if not entries:
            return []

        if not keywords:
            keywords = extract_keywords_simple(task_desc)
        if not keywords:
            return entries[: self.top_k]

        if self.embedding_model:
            return self._retrieve_embedding(entries, keywords)

        return self._retrieve_tf(entries, keywords)

    def _retrieve_embedding(self, entries: list[MemoryEntry], keywords: list) -> list:
        """AveFact with neural embeddings (paper algorithm)."""
        self._ensure_entry_embeddings(entries)

        # Embed all keywords in a single batch call
        kw_vectors = self._embed_batch(keywords)

        scored = []
        for entry in entries:
            entry_vec = self._emb_cache.get(entry.id)
            if entry_vec is None:
                continue
            per_kw = [_cosine_vec(kw_vec, entry_vec) for kw_vec in kw_vectors]
            avg = sum(per_kw) / len(per_kw)
            scored.append((avg, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[: self.top_k]]

    def _retrieve_tf(self, entries: list[MemoryEntry], keywords: list) -> list:
        """AveFact with TF-weighted token overlap (fallback, no API needed)."""
        scored = []
        for entry in entries:
            doc_tokens = (
                [kw.lower() for kw in entry.keywords]
                + _tokenize(entry.task_desc)
                + _tokenize(entry.script)
            )
            per_kw = [_cosine_tf(_tokenize(kw), doc_tokens) for kw in keywords]
            avg = sum(per_kw) / len(per_kw)
            scored.append((avg, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[: self.top_k]]
