"""Injected capability protocols for the search subsystem.

These are the ports that the search layer depends on. Concrete
implementations are injected at startup — the search layer never
imports a specific vector store, LLM, or cache backend directly.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class VectorStorePort(Protocol):
    """Vector similarity search capability."""

    async def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        namespace: str = "",
    ) -> list[dict[str, Any]]:
        """Return top_k nearest neighbors with scores and metadata."""
        ...

    async def upsert(
        self,
        documents: list[dict[str, Any]],
        *,
        namespace: str = "",
    ) -> int:
        """Upsert documents with embeddings. Return count inserted."""
        ...


@runtime_checkable
class EmbeddingPort(Protocol):
    """Text embedding capability."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for input texts."""
        ...

    @property
    def dimensions(self) -> int:
        """Return the dimensionality of the embedding model."""
        ...


@runtime_checkable
class CompletionPort(Protocol):
    """LLM completion capability."""

    async def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Return a completion for the given prompt."""
        ...


@runtime_checkable
class CachePort(Protocol):
    """Cache read/write capability."""

    async def get(self, key: str) -> Any | None:
        """Return cached value or None."""
        ...

    async def set(self, key: str, value: Any, *, ttl: int = 3600) -> None:
        """Cache a value with TTL in seconds."""
        ...


@runtime_checkable
class RerankerPort(Protocol):
    """Reranking capability."""

    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Return (original_index, score) pairs sorted by relevance."""
        ...


@runtime_checkable
class KeywordSearchPort(Protocol):
    """BM25 / keyword search capability."""

    async def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        namespace: str = "",
    ) -> list[dict[str, Any]]:
        """Return top_k keyword search results with scores."""
        ...
