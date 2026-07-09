"""Tokenizer and stop words utilities for the search subsystem."""

from __future__ import annotations

import re
from functools import lru_cache

STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "this", "that",
    "these", "those", "it", "its", "i", "me", "my", "we", "our", "you",
    "your", "he", "him", "his", "she", "her", "they", "them", "their",
    "what", "which", "who", "whom", "when", "where", "why", "how",
    "not", "no", "nor", "so", "if", "then", "than", "too", "very",
    "just", "about", "above", "after", "again", "all", "also", "am",
    "any", "as", "because", "before", "between", "both", "each",
    "few", "more", "most", "other", "over", "own", "same", "some",
    "such", "into", "only", "out", "up", "down",
})

_TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase word tokens."""
    return _TOKEN_PATTERN.findall(text.lower())


def remove_stop_words(tokens: list[str]) -> list[str]:
    """Remove English stop words from token list."""
    return [t for t in tokens if t not in STOP_WORDS]


@lru_cache(maxsize=4096)
def normalize_query(query: str) -> str:
    """Normalize a query: lowercase, tokenize, remove stops, rejoin."""
    tokens = tokenize(query)
    filtered = remove_stop_words(tokens)
    return " ".join(filtered) if filtered else query.lower().strip()
