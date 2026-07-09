"""BM25 keyword retriever using the KeywordSearchPort."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..ports import KeywordSearchPort


@dataclass(frozen=True)
class BM25Config:
    k1: float = 1.5
    b: float = 0.75
    top_k: int = 10


class BM25Retriever:
    """BM25 retrieval via injected keyword search port."""

    def __init__(self, keyword_search: KeywordSearchPort, config: BM25Config | None = None) -> None:
        self._search = keyword_search
        self._config = config or BM25Config()

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        namespace: str = "",
    ) -> list[dict[str, Any]]:
        k = top_k or self._config.top_k
        return await self._search.search(query, top_k=k, filters=filters, namespace=namespace)
