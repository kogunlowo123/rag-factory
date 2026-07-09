"""Query classification — routes queries to optimal retrieval strategy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..ports import CompletionPort
from ...contracts import RetrievalStrategy

_CLASSIFY_PROMPT = """Classify this search query into ONE category:
- FACTUAL: simple fact lookup ("What is EC2?")
- ANALYTICAL: comparison or analysis ("AWS vs Azure pricing")
- PROCEDURAL: how-to or step-by-step ("How to deploy Lambda")
- EXPLORATORY: broad topic exploration ("cloud security best practices")
- MULTI_HOP: requires combining multiple sources ("impact of serverless on cost and latency")

Query: {query}

Return ONLY the category name, nothing else."""


class QueryType(str, Enum):
    FACTUAL = "FACTUAL"
    ANALYTICAL = "ANALYTICAL"
    PROCEDURAL = "PROCEDURAL"
    EXPLORATORY = "EXPLORATORY"
    MULTI_HOP = "MULTI_HOP"


@dataclass(frozen=True)
class ClassificationResult:
    query_type: QueryType
    recommended_retrieval: RetrievalStrategy
    confidence: float


# Strategy mapping: query type → best retrieval strategy
_STRATEGY_MAP: dict[QueryType, RetrievalStrategy] = {
    QueryType.FACTUAL: RetrievalStrategy.DENSE,
    QueryType.ANALYTICAL: RetrievalStrategy.HYBRID,
    QueryType.PROCEDURAL: RetrievalStrategy.HYBRID,
    QueryType.EXPLORATORY: RetrievalStrategy.HYBRID,
    QueryType.MULTI_HOP: RetrievalStrategy.GRAPH,
}


class QueryClassifier:
    """Classify queries and recommend retrieval strategies."""

    def __init__(self, completion: CompletionPort) -> None:
        self._completion = completion

    async def classify(self, query: str) -> ClassificationResult:
        response = await self._completion.complete(
            _CLASSIFY_PROMPT.format(query=query),
            system="You are a query classifier. Return only the category name.",
            temperature=0.0,
            max_tokens=20,
        )

        raw = response.strip().upper()
        try:
            query_type = QueryType(raw)
        except ValueError:
            query_type = QueryType.ANALYTICAL

        return ClassificationResult(
            query_type=query_type,
            recommended_retrieval=_STRATEGY_MAP[query_type],
            confidence=0.85,
        )
