"""Exact-match cache — in-memory LRU for deterministic query hits."""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass
class CacheEntry:
    value: Any
    expires_at: float


class ExactCache:
    """In-memory LRU cache with TTL. Thread-safe for single-process use."""

    def __init__(self, max_size: int = 1024, default_ttl: int = 3600) -> None:
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl

    async def get(self, key: str) -> Any | None:
        normalized = self._normalize_key(key)
        entry = self._store.get(normalized)
        if entry is None:
            return None
        if time.monotonic() > entry.expires_at:
            del self._store[normalized]
            return None
        self._store.move_to_end(normalized)
        return entry.value

    async def set(self, key: str, value: Any, *, ttl: int | None = None) -> None:
        normalized = self._normalize_key(key)
        effective_ttl = ttl if ttl is not None else self._default_ttl
        self._store[normalized] = CacheEntry(
            value=value,
            expires_at=time.monotonic() + effective_ttl,
        )
        self._store.move_to_end(normalized)
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)

    async def invalidate(self, key: str) -> None:
        normalized = self._normalize_key(key)
        self._store.pop(normalized, None)

    async def clear(self) -> None:
        self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)

    @staticmethod
    def _normalize_key(key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()
