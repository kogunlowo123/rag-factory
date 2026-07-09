"""Tests for RAG contracts — enums, dataclasses, immutability."""

from app.contracts import (
    CacheStrategy,
    QueryStrategy,
    RerankStrategy,
    RetrievalStrategy,
    SearchRequest,
    SearchResponse,
    SearchResult,
    Tenant,
)


class TestTenant:
    def test_frozen(self):
        t = Tenant(id="t1", name="Test", namespace="test")
        assert t.id == "t1"
        # frozen dataclass — assignment should raise
        try:
            t.id = "t2"  # type: ignore[misc]
            assert False, "Should have raised"
        except AttributeError:
            pass

    def test_metadata_default(self):
        t = Tenant(id="t1", name="Test", namespace="ns")
        assert t.metadata == {}


class TestSearchRequest:
    def test_defaults(self):
        tenant = Tenant(id="t1", name="Test", namespace="ns")
        req = SearchRequest(query="what is cloud?", tenant=tenant)
        assert req.top_k == 10
        assert req.retrieval == RetrievalStrategy.HYBRID
        assert req.rerank == RerankStrategy.CROSS_ENCODER
        assert req.query_strategy == QueryStrategy.CLASSIFY
        assert req.cache == CacheStrategy.SEMANTIC

    def test_custom_strategies(self):
        tenant = Tenant(id="t1", name="Test", namespace="ns")
        req = SearchRequest(
            query="what is terraform?",
            tenant=tenant,
            retrieval=RetrievalStrategy.BM25,
            rerank=RerankStrategy.NONE,
            top_k=5,
        )
        assert req.retrieval == RetrievalStrategy.BM25
        assert req.rerank == RerankStrategy.NONE
        assert req.top_k == 5


class TestSearchResponse:
    def test_construction(self):
        resp = SearchResponse(
            results=[
                SearchResult(id="1", content="doc1", score=0.95, source="s3://bucket/doc1"),
                SearchResult(id="2", content="doc2", score=0.87),
            ],
            query="test query",
            strategy_used=RetrievalStrategy.HYBRID,
            rerank_used=RerankStrategy.CROSS_ENCODER,
            latency_ms=42.5,
        )
        assert len(resp.results) == 2
        assert resp.results[0].score == 0.95
        assert resp.cache_hit is False


class TestEnums:
    def test_retrieval_values(self):
        assert RetrievalStrategy.BM25.value == "bm25"
        assert RetrievalStrategy.HYBRID.value == "hybrid"
        assert RetrievalStrategy.GRAPH.value == "graph"

    def test_rerank_values(self):
        assert RerankStrategy.CROSS_ENCODER.value == "cross_encoder"
        assert RerankStrategy.CASCADE.value == "cascade"

    def test_query_values(self):
        assert QueryStrategy.HYDE.value == "hyde"
        assert QueryStrategy.DECOMPOSE.value == "decompose"
