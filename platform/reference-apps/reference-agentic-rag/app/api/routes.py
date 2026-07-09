"""API routes for the RAG search service."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..contracts import (
    CacheStrategy,
    QueryStrategy,
    RerankStrategy,
    RetrievalStrategy,
    SearchRequest,
    Tenant,
)

router = APIRouter(prefix="/api/v1", tags=["search"])


class SearchRequestBody(BaseModel):
    """API request body for search."""
    query: str = Field(..., min_length=1, max_length=2000)
    tenant_id: str = Field(default="default")
    top_k: int = Field(default=10, ge=1, le=100)
    retrieval: RetrievalStrategy = RetrievalStrategy.HYBRID
    rerank: RerankStrategy = RerankStrategy.CROSS_ENCODER
    query_strategy: QueryStrategy = QueryStrategy.CLASSIFY
    cache: CacheStrategy = CacheStrategy.SEMANTIC
    filters: dict = Field(default_factory=dict)


class SearchResultItem(BaseModel):
    """Single search result in API response."""
    id: str
    content: str
    score: float
    metadata: dict = Field(default_factory=dict)
    source: str = ""


class SearchResponseBody(BaseModel):
    """API response body for search."""
    results: list[SearchResultItem]
    query: str
    strategy_used: str
    rerank_used: str
    latency_ms: float
    cache_hit: bool
    total: int


class IngestRequestBody(BaseModel):
    """API request body for document ingestion."""
    documents: list[dict] = Field(..., min_length=1)
    namespace: str = ""


class IngestResponseBody(BaseModel):
    """API response body for ingestion."""
    ingested: int
    namespace: str


@router.post("/search", response_model=SearchResponseBody)
async def search(body: SearchRequestBody) -> SearchResponseBody:
    """Execute an adaptive search across the knowledge base."""
    # Router is injected via app state at startup
    from ..main import app

    adaptive_router = getattr(app.state, "router", None)
    if not adaptive_router:
        raise HTTPException(
            status_code=503,
            detail="Search router not initialized. Configure vector store and embedding ports.",
        )

    tenant = Tenant(id=body.tenant_id, name=body.tenant_id, namespace=body.tenant_id)
    request = SearchRequest(
        query=body.query,
        tenant=tenant,
        top_k=body.top_k,
        retrieval=body.retrieval,
        rerank=body.rerank,
        query_strategy=body.query_strategy,
        cache=body.cache,
        filters=body.filters,
    )

    response = await adaptive_router.search(request)

    return SearchResponseBody(
        results=[
            SearchResultItem(
                id=r.id,
                content=r.content,
                score=r.score,
                metadata=r.metadata,
                source=r.source,
            )
            for r in response.results
        ],
        query=response.query,
        strategy_used=response.strategy_used.value,
        rerank_used=response.rerank_used.value,
        latency_ms=response.latency_ms,
        cache_hit=response.cache_hit,
        total=len(response.results),
    )


@router.post("/ingest", response_model=IngestResponseBody)
async def ingest(body: IngestRequestBody) -> IngestResponseBody:
    """Ingest documents into the vector store."""
    from ..main import app

    vector_store = getattr(app.state, "vector_store", None)
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized.")

    count = await vector_store.upsert(body.documents, namespace=body.namespace)
    return IngestResponseBody(ingested=count, namespace=body.namespace)


@router.get("/strategies")
async def list_strategies() -> dict:
    """List all available retrieval, reranking, and query strategies."""
    return {
        "retrieval": [s.value for s in RetrievalStrategy],
        "reranking": [s.value for s in RerankStrategy],
        "query": [s.value for s in QueryStrategy],
        "cache": [s.value for s in CacheStrategy],
    }
