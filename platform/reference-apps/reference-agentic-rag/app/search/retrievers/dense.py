"""Dense (embedding-based) retriever using VectorStorePort + EmbeddingPort."""

from __future__ import annotations

from typing import Any

from ..ports import EmbeddingPort, VectorStorePort


class DenseRetriever:
    """Dense retrieval via embedding similarity search."""

    def __init__(self, vector_store: VectorStorePort, embedding: EmbeddingPort) -> None:
        self._store = vector_store
        self._embedding = embedding

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        namespace: str = "",
    ) -> list[dict[str, Any]]:
        query_embedding = (await self._embedding.embed([query]))[0]
        return await self._store.search(
            query_embedding, top_k=top_k, filters=filters, namespace=namespace
        )
