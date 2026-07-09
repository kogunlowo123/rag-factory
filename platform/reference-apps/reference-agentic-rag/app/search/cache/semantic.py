"""Semantic cache — find cached results for semantically similar queries."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..ports import EmbeddingPort


@dataclass
class SemanticEntry:
    query: str
    embedding: list[float]
    value: Any
    expires_at: float


class SemanticCache:
    """Cache that matches semantically similar queries via cosine similarity."""

    def __init__(
        self,
        embedding: EmbeddingPort,
        *,
        similarity_threshold: float = 0.95,
        max_size: int = 512,
        default_ttl: int = 3600,
    ) -> None:
        self._embedding = embedding
        self._threshold = similarity_threshold
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._entries: list[SemanticEntry] = []

    async def get(self, key: str) -> Any | None:
        if not self._entries:
            return None

        query_emb = (await self._embedding.embed([key]))[0]
        now = time.monotonic()

        best_score = 0.0
        best_entry: SemanticEntry | None = None

        alive: list[SemanticEntry] = []
        for entry in self._entries:
            if now > entry.expires_at:
                continue
            alive.append(entry)
            score = self._cosine_similarity(query_emb, entry.embedding)
            if score > best_score:
                best_score = score
                best_entry = entry

        self._entries = alive

        if best_entry and best_score >= self._threshold:
            return best_entry.value
        return None

    async def set(self, key: str, value: Any, *, ttl: int | None = None) -> None:
        query_emb = (await self._embedding.embed([key]))[0]
        effective_ttl = ttl if ttl is not None else self._default_ttl

        self._entries.append(
            SemanticEntry(
                query=key,
                embedding=query_emb,
                value=value,
                expires_at=time.monotonic() + effective_ttl,
            )
        )

        if len(self._entries) > self._max_size:
            self._entries = self._entries[-self._max_size:]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
