"""HyDE — Hypothetical Document Embeddings.

Generate a hypothetical answer, embed it, and use that embedding
for retrieval. Often outperforms raw query embedding on technical content.
"""

from __future__ import annotations

from ..ports import CompletionPort, EmbeddingPort

_HYDE_PROMPT = """Write a short, factual paragraph that would answer this question.
Write as if you are writing a technical document. Be specific and detailed.

Question: {query}"""


class HyDEGenerator:
    """Generate hypothetical documents for improved dense retrieval."""

    def __init__(self, completion: CompletionPort, embedding: EmbeddingPort) -> None:
        self._completion = completion
        self._embedding = embedding

    async def generate_embedding(self, query: str) -> list[float]:
        """Generate a HyDE embedding: query → hypothetical doc → embedding."""
        hypothetical = await self._completion.complete(
            _HYDE_PROMPT.format(query=query),
            system="You are a technical writer. Write a factual, specific paragraph.",
            temperature=0.3,
            max_tokens=300,
        )
        embeddings = await self._embedding.embed([hypothetical])
        return embeddings[0]

    async def generate_document(self, query: str) -> str:
        """Generate just the hypothetical document (for inspection/debugging)."""
        return await self._completion.complete(
            _HYDE_PROMPT.format(query=query),
            system="You are a technical writer. Write a factual, specific paragraph.",
            temperature=0.3,
            max_tokens=300,
        )
