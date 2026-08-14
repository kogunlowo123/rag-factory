"""Cross-encoder reranking via RerankerPort."""

from __future__ import annotations

from ...contracts import SearchResult
from ..ports import RerankerPort


class CrossEncoderReranker:
    """Rerank results using a cross-encoder model (via injected port)."""

    def __init__(self, reranker: RerankerPort) -> None:
        self._reranker = reranker

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        *,
        top_k: int = 10,
    ) -> list[SearchResult]:
        if not results:
            return results

        documents = [r.content for r in results]
        scored = await self._reranker.rerank(query, documents, top_k=top_k)

        return [
            SearchResult(
                id=results[idx].id,
                content=results[idx].content,
                score=score,
                metadata={**results[idx].metadata, "original_score": results[idx].score},
                source=results[idx].source,
            )
            for idx, score in scored
        ]
