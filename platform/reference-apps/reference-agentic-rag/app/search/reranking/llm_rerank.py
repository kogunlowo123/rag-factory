"""LLM-based reranking — asks the completion model to score relevance."""

from __future__ import annotations

import json

from ...contracts import SearchResult
from ..ports import CompletionPort

_RERANK_PROMPT = """Score each document's relevance to the query on a scale of 0-10.
Return ONLY a JSON array of objects: [{{"index": 0, "score": 8.5}}, ...]

Query: {query}

Documents:
{documents}"""


class LLMReranker:
    """Rerank results by asking an LLM to score relevance."""

    def __init__(self, completion: CompletionPort) -> None:
        self._completion = completion

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        *,
        top_k: int = 10,
    ) -> list[SearchResult]:
        if not results:
            return results

        doc_text = "\n".join(
            f"[{i}] {r.content[:500]}" for i, r in enumerate(results)
        )

        response = await self._completion.complete(
            _RERANK_PROMPT.format(query=query, documents=doc_text),
            system="You are a relevance scoring assistant. Return only valid JSON.",
            temperature=0.0,
            max_tokens=512,
        )

        try:
            scores = json.loads(response.strip().removeprefix("```json").removesuffix("```"))
            scored_pairs = [(s["index"], s["score"]) for s in scores]
            scored_pairs.sort(key=lambda x: x[1], reverse=True)
        except (json.JSONDecodeError, KeyError, TypeError):
            return results[:top_k]

        return [
            SearchResult(
                id=results[idx].id,
                content=results[idx].content,
                score=score / 10.0,
                metadata={**results[idx].metadata, "llm_relevance": score},
                source=results[idx].source,
            )
            for idx, score in scored_pairs[:top_k]
            if idx < len(results)
        ]
