"""RAG contracts — enums, tenancy, models, retrieval types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RetrievalStrategy(str, Enum):
    """Supported retrieval strategies."""
    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"
    GRAPH = "graph"
    PARENT_DOC = "parent_doc"


class RerankStrategy(str, Enum):
    """Supported reranking strategies."""
    CROSS_ENCODER = "cross_encoder"
    LLM_RERANK = "llm_rerank"
    BOOST = "boost"
    CASCADE = "cascade"
    NONE = "none"


class QueryStrategy(str, Enum):
    """Query preprocessing strategies."""
    PASSTHROUGH = "passthrough"
    CLASSIFY = "classify"
    REWRITE = "rewrite"
    EXPAND = "expand"
    HYDE = "hyde"
    DECOMPOSE = "decompose"


class CacheStrategy(str, Enum):
    """Cache strategies."""
    NONE = "none"
    EXACT = "exact"
    SEMANTIC = "semantic"
    REDIS = "redis"


@dataclass(frozen=True)
class Tenant:
    """Tenant identity for multi-tenant isolation."""
    id: str
    name: str
    namespace: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchRequest:
    """Inbound search request contract."""
    query: str
    tenant: Tenant
    top_k: int = 10
    retrieval: RetrievalStrategy = RetrievalStrategy.HYBRID
    rerank: RerankStrategy = RerankStrategy.CROSS_ENCODER
    query_strategy: QueryStrategy = QueryStrategy.CLASSIFY
    cache: CacheStrategy = CacheStrategy.SEMANTIC
    filters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    """Single search result."""
    id: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = ""


@dataclass(frozen=True)
class SearchResponse:
    """Outbound search response contract."""
    results: list[SearchResult]
    query: str
    strategy_used: RetrievalStrategy
    rerank_used: RerankStrategy
    latency_ms: float
    cache_hit: bool = False
