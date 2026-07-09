"""RAG generation chain — search → format context → generate answer."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..contracts import SearchRequest, SearchResponse, Tenant
from ..search.adaptive.router import AdaptiveRouter
from ..search.ports import CompletionPort

_RAG_SYSTEM = """You are a knowledgeable assistant for cloud computing education.
Answer the question using ONLY the provided context. If the context doesn't contain
enough information, say so. Cite specific sources when possible."""

_RAG_PROMPT = """Context:
{context}

Question: {question}

Provide a clear, accurate answer based on the context above."""


@dataclass(frozen=True)
class RAGResponse:
    answer: str
    sources: list[str]
    search_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    retrieval_strategy: str
    cache_hit: bool


class RAGChain:
    """End-to-end RAG: query → retrieve → generate."""

    def __init__(
        self,
        router: AdaptiveRouter,
        completion: CompletionPort,
        *,
        max_context_tokens: int = 4000,
    ) -> None:
        self._router = router
        self._completion = completion
        self._max_context = max_context_tokens

    async def answer(
        self,
        question: str,
        *,
        tenant_id: str = "default",
        top_k: int = 5,
    ) -> RAGResponse:
        import time

        start = time.monotonic()

        # 1. Retrieve
        tenant = Tenant(id=tenant_id, name=tenant_id, namespace=tenant_id)
        request = SearchRequest(query=question, tenant=tenant, top_k=top_k)
        search_response: SearchResponse = await self._router.search(request)
        search_elapsed = time.monotonic() - start

        # 2. Format context
        context_parts: list[str] = []
        sources: list[str] = []
        char_budget = self._max_context * 4  # rough chars-to-tokens

        for result in search_response.results:
            if len("\n".join(context_parts)) > char_budget:
                break
            context_parts.append(f"[{result.source or result.id}]: {result.content}")
            if result.source:
                sources.append(result.source)

        context = "\n\n".join(context_parts) if context_parts else "No relevant context found."

        # 3. Generate
        gen_start = time.monotonic()
        answer = await self._completion.complete(
            _RAG_PROMPT.format(context=context, question=question),
            system=_RAG_SYSTEM,
            temperature=0.1,
            max_tokens=1024,
        )
        gen_elapsed = time.monotonic() - gen_start
        total = time.monotonic() - start

        return RAGResponse(
            answer=answer.strip(),
            sources=sources,
            search_latency_ms=search_elapsed * 1000,
            generation_latency_ms=gen_elapsed * 1000,
            total_latency_ms=total * 1000,
            retrieval_strategy=search_response.strategy_used.value,
            cache_hit=search_response.cache_hit,
        )
