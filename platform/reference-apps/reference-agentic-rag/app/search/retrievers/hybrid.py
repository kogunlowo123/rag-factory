"""Hybrid retriever — BM25 + Dense with reciprocal rank fusion."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from ..ports import EmbeddingPort, KeywordSearchPort, VectorStorePort


@dataclass(frozen=True)
class HybridConfig:
    rrf_k: int = 60
    dense_weight: float = 0.6
    bm25_weight: float = 0.4
    oversample: int = 2


class HybridRetriever:
    """Reciprocal rank fusion of BM25 + dense retrieval."""

    def __init__(
        self,
        vector_store: VectorStorePort,
        embedding: EmbeddingPort,
        keyword_search: KeywordSearchPort,
        config: HybridConfig | None = None,
    ) -> None:
        self._store = vector_store
        self._embedding = embedding
        self._keyword = keyword_search
        self._config = config or HybridConfig()

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        namespace: str = "",
    ) -> list[dict[str, Any]]:
        fetch_k = top_k * self._config.oversample
        query_embedding = (await self._embedding.embed([query]))[0]

        dense_task = self._store.search(
            query_embedding, top_k=fetch_k, filters=filters, namespace=namespace
        )
        keyword_task = self._keyword.search(
            query, top_k=fetch_k, filters=filters, namespace=namespace
        )

        dense_results, keyword_results = await asyncio.gather(dense_task, keyword_task)

        fused = self._reciprocal_rank_fusion(
            dense_results,
            keyword_results,
            k=self._config.rrf_k,
            weights=(self._config.dense_weight, self._config.bm25_weight),
        )
        return fused[:top_k]

    @staticmethod
    def _reciprocal_rank_fusion(
        *result_lists: list[dict[str, Any]],
        k: int = 60,
        weights: tuple[float, ...] | None = None,
    ) -> list[dict[str, Any]]:
        if weights is None:
            weights = tuple(1.0 for _ in result_lists)

        scores: dict[str, float] = {}
        docs: dict[str, dict[str, Any]] = {}

        for weight, results in zip(weights, result_lists, strict=False):
            for rank, doc in enumerate(results):
                doc_id = doc.get("id", str(rank))
                scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank + 1)
                docs[doc_id] = doc

        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
        return [{**docs[did], "score": scores[did]} for did in sorted_ids]
