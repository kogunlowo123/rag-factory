"""Tests for retrievers using mock ports."""

from unittest.mock import AsyncMock

from app.search.retrievers.dense import DenseRetriever
from app.search.retrievers.bm25 import BM25Retriever, BM25Config
from app.search.retrievers.hybrid import HybridRetriever


class TestDenseRetriever:
    async def test_retrieve(self):
        mock_store = AsyncMock()
        mock_store.search.return_value = [
            {"id": "1", "content": "doc1", "score": 0.9},
            {"id": "2", "content": "doc2", "score": 0.8},
        ]
        mock_embedding = AsyncMock()
        mock_embedding.embed.return_value = [[0.1, 0.2, 0.3]]

        retriever = DenseRetriever(mock_store, mock_embedding)
        results = await retriever.retrieve("test query", top_k=2)

        assert len(results) == 2
        mock_embedding.embed.assert_called_once_with(["test query"])


class TestBM25Retriever:
    async def test_retrieve(self):
        mock_search = AsyncMock()
        mock_search.search.return_value = [
            {"id": "1", "content": "keyword match", "score": 5.2},
        ]

        retriever = BM25Retriever(mock_search, BM25Config(top_k=5))
        results = await retriever.retrieve("keyword query")

        assert len(results) == 1


class TestHybridRetriever:
    async def test_fusion(self):
        mock_store = AsyncMock()
        mock_store.search.return_value = [
            {"id": "a", "content": "dense hit", "score": 0.9},
            {"id": "b", "content": "both", "score": 0.7},
        ]
        mock_embedding = AsyncMock()
        mock_embedding.embed.return_value = [[0.1, 0.2]]
        mock_keyword = AsyncMock()
        mock_keyword.search.return_value = [
            {"id": "b", "content": "both", "score": 4.1},
            {"id": "c", "content": "keyword only", "score": 3.5},
        ]

        retriever = HybridRetriever(mock_store, mock_embedding, mock_keyword)
        results = await retriever.retrieve("test", top_k=3)

        ids = [r["id"] for r in results]
        assert ids[0] == "b"
        assert len(results) == 3
