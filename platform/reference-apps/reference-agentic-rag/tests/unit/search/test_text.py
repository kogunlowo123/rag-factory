"""Tests for text utilities — tokenizer and stop words."""

from app.search._text import normalize_query, remove_stop_words, tokenize


class TestTokenize:
    def test_basic(self):
        assert tokenize("Hello World") == ["hello", "world"]

    def test_punctuation(self):
        tokens = tokenize("What is AWS? It's great!")
        assert "what" in tokens
        assert "aws" in tokens

    def test_empty(self):
        assert tokenize("") == []


class TestStopWords:
    def test_removes_stops(self):
        tokens = ["what", "is", "the", "best", "cloud"]
        filtered = remove_stop_words(tokens)
        assert "what" not in filtered
        assert "is" not in filtered
        assert "best" in filtered
        assert "cloud" in filtered


class TestNormalizeQuery:
    def test_basic(self):
        result = normalize_query("What is the best cloud certification?")
        assert "best" in result
        assert "cloud" in result
        assert "certification" in result
        assert "what" not in result
        assert "is" not in result
        assert "the" not in result

    def test_caching(self):
        r1 = normalize_query("test query")
        r2 = normalize_query("test query")
        assert r1 == r2
