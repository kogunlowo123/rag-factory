"""Adaptive router — the public facade for the search subsystem.

Routes incoming SearchRequests to the appropriate retrieval pipeline
based on query classification and tenant configuration.
"""

from __future__ import annotations

import time
from dataclasses import replace

from ..ports import (
    CachePort,
    CompletionPort,
    EmbeddingPort,
    KeywordSearchPort,
    RerankerPort,
    VectorStorePort,
)
from ...contracts import (
    CacheStrategy,
    QueryStrategy,
    RerankStrategy,
    RetrievalStrategy,
    SearchRequest,
    SearchResponse,
    SearchResult,
)


class AdaptiveRouter:
    """Route search requests to the best retrieval pipeline.

    The router:
    1. Checks the cache (if enabled)
    2. Classifies or rewrites the query (if configured)
    3. Dispatches to the correct retriever (bm25, dense, hybrid, graph, parent_doc)
    4. Reranks results (if configured)
    5. Caches the response (if enabled)
    """

    def __init__(
        self,
        *,
        vector_store: VectorStorePort,
        embedding: EmbeddingPort,
        completion: CompletionPort,
        keyword_search: KeywordSearchPort | None = None,
        reranker: RerankerPort | None = None,
        cache: CachePort | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._embedding = embedding
        self._completion = completion
        self._keyword_search = keyword_search
        self._reranker = reranker
        self._cache = cache

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Execute a search request through the adaptive pipeline."""
        start = time.monotonic()

        # 1. Cache check
        if request.cache != CacheStrategy.NONE and self._cache:
            cached = await self._check_cache(request)
            if cached is not None:
                elapsed = (time.monotonic() - start) * 1000
                return replace(cached, latency_ms=elapsed, cache_hit=True)

        # 2. Query preprocessing
        processed_query = await self._preprocess_query(request)

        # 3. Retrieval
        raw_results = await self._retrieve(
            query=processed_query,
            request=request,
        )

        # 4. Reranking
        if request.rerank != RerankStrategy.NONE and self._reranker:
            raw_results = await self._rerank(
                query=processed_query,
                results=raw_results,
                strategy=request.rerank,
                top_k=request.top_k,
            )

        # 5. Build response
        elapsed = (time.monotonic() - start) * 1000
        response = SearchResponse(
            results=raw_results[: request.top_k],
            query=processed_query,
            strategy_used=request.retrieval,
            rerank_used=request.rerank,
            latency_ms=elapsed,
        )

        # 6. Cache store
        if request.cache != CacheStrategy.NONE and self._cache:
            await self._store_cache(request, response)

        return response

    async def _check_cache(self, request: SearchRequest) -> SearchResponse | None:
        """Check cache for a previous response."""
        if not self._cache:
            return None
        key = f"search:{request.tenant.id}:{request.query}"
        return await self._cache.get(key)

    async def _store_cache(
        self, request: SearchRequest, response: SearchResponse
    ) -> None:
        """Store response in cache."""
        if not self._cache:
            return
        key = f"search:{request.tenant.id}:{request.query}"
        await self._cache.set(key, response, ttl=3600)

    async def _preprocess_query(self, request: SearchRequest) -> str:
        """Apply query preprocessing strategy."""
        if request.query_strategy == QueryStrategy.PASSTHROUGH:
            return request.query

        if request.query_strategy == QueryStrategy.REWRITE:
            return await self._completion.complete(
                f"Rewrite this search query for better retrieval:\n\n{request.query}",
                system="You are a query rewriting assistant. Return only the rewritten query.",
                max_tokens=200,
            )

        # Default: passthrough for unimplemented strategies
        return request.query

    async def _retrieve(
        self,
        *,
        query: str,
        request: SearchRequest,
    ) -> list[SearchResult]:
        """Dispatch to the appropriate retriever."""
        embedding = (await self._embedding.embed([query]))[0]

        if request.retrieval == RetrievalStrategy.BM25:
            if not self._keyword_search:
                raise ValueError("BM25 retrieval requires keyword_search port")
            raw = await self._keyword_search.search(
                query, top_k=request.top_k, namespace=request.tenant.namespace
            )
        elif request.retrieval == RetrievalStrategy.DENSE:
            raw = await self._vector_store.search(
                embedding, top_k=request.top_k, namespace=request.tenant.namespace
            )
        elif request.retrieval == RetrievalStrategy.HYBRID:
            raw = await self._hybrid_retrieve(
                query=query,
                embedding=embedding,
                top_k=request.top_k,
                namespace=request.tenant.namespace,
            )
        else:
            # Fallback to dense for unimplemented strategies
            raw = await self._vector_store.search(
                embedding, top_k=request.top_k, namespace=request.tenant.namespace
            )

        return [
            SearchResult(
                id=r.get("id", ""),
                content=r.get("content", ""),
                score=r.get("score", 0.0),
                metadata=r.get("metadata", {}),
                source=r.get("source", ""),
            )
            for r in raw
        ]

    async def _hybrid_retrieve(
        self,
        *,
        query: str,
        embedding: list[float],
        top_k: int,
        namespace: str,
    ) -> list[dict]:
        """Reciprocal rank fusion of BM25 + dense retrieval."""
        dense_results = await self._vector_store.search(
            embedding, top_k=top_k * 2, namespace=namespace
        )

        if self._keyword_search:
            keyword_results = await self._keyword_search.search(
                query, top_k=top_k * 2, namespace=namespace
            )
        else:
            keyword_results = []

        return self._reciprocal_rank_fusion(dense_results, keyword_results, k=60)

    @staticmethod
    def _reciprocal_rank_fusion(
        *result_lists: list[dict],
        k: int = 60,
    ) -> list[dict]:
        """Merge multiple result lists using reciprocal rank fusion."""
        scores: dict[str, float] = {}
        docs: dict[str, dict] = {}

        for results in result_lists:
            for rank, doc in enumerate(results):
                doc_id = doc.get("id", str(rank))
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
                docs[doc_id] = doc

        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
        return [
            {**docs[doc_id], "score": scores[doc_id]}
            for doc_id in sorted_ids
        ]

    async def _rerank(
        self,
        *,
        query: str,
        results: list[SearchResult],
        strategy: RerankStrategy,
        top_k: int,
    ) -> list[SearchResult]:
        """Apply reranking strategy to results."""
        if not self._reranker or not results:
            return results

        documents = [r.content for r in results]
        reranked = await self._reranker.rerank(query, documents, top_k=top_k)

        return [
            SearchResult(
                id=results[idx].id,
                content=results[idx].content,
                score=score,
                metadata=results[idx].metadata,
                source=results[idx].source,
            )
            for idx, score in reranked
        ]
