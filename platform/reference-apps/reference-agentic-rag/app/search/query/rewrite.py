"""Query rewriting — reformulate queries for better retrieval."""

from __future__ import annotations

from ..ports import CompletionPort

_REWRITE_PROMPT = """Rewrite this search query to improve retrieval quality.
Make it more specific, add relevant technical terms, and expand abbreviations.
Return ONLY the rewritten query, nothing else.

Original query: {query}"""


class QueryRewriter:
    """Rewrite queries for improved retrieval."""

    def __init__(self, completion: CompletionPort) -> None:
        self._completion = completion

    async def rewrite(self, query: str) -> str:
        response = await self._completion.complete(
            _REWRITE_PROMPT.format(query=query),
            system="You are a query rewriting assistant. Return only the rewritten query.",
            temperature=0.0,
            max_tokens=200,
        )
        rewritten = response.strip().strip('"').strip("'")
        return rewritten if rewritten else query
