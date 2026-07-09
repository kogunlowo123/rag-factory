"""Tests for exact cache implementation."""

import pytest
from app.search.cache.exact import ExactCache


class TestExactCache:
    @pytest.fixture
    def cache(self):
        return ExactCache(max_size=5, default_ttl=3600)

    async def test_get_miss(self, cache):
        result = await cache.get("nonexistent")
        assert result is None

    async def test_set_and_get(self, cache):
        await cache.set("key1", {"data": "value"})
        result = await cache.get("key1")
        assert result == {"data": "value"}

    async def test_eviction(self, cache):
        for i in range(7):
            await cache.set(f"key{i}", f"val{i}")
        assert cache.size <= 5

    async def test_invalidate(self, cache):
        await cache.set("key1", "value")
        await cache.invalidate("key1")
        result = await cache.get("key1")
        assert result is None

    async def test_clear(self, cache):
        await cache.set("a", 1)
        await cache.set("b", 2)
        await cache.clear()
        assert cache.size == 0
