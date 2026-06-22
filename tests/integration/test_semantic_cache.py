"""Integration tests for the semantic LLM cache."""

import infino
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.outputs import Generation

from langchain_infino import InfinoSemanticCache

EMBED_DIM = 16


@pytest.fixture
def cache(tmp_path) -> InfinoSemanticCache:
    connection = infino.connect(str(tmp_path / "cache_db"))
    return InfinoSemanticCache(
        connection, DeterministicFakeEmbedding(size=EMBED_DIM), dim=EMBED_DIM
    )


def test_cache_miss_on_empty(cache: InfinoSemanticCache) -> None:
    assert cache.lookup("hello world", "gpt-x") is None


def test_cache_hit_after_update(cache: InfinoSemanticCache) -> None:
    cache.update("what is the capital of France", "gpt-x", [Generation(text="Paris")])
    hit = cache.lookup("what is the capital of France", "gpt-x")
    assert hit is not None
    assert hit[0].text == "Paris"


def test_cache_miss_on_different_model(cache: InfinoSemanticCache) -> None:
    cache.update("ping", "gpt-x", [Generation(text="pong")])
    assert cache.lookup("ping", "gpt-y") is None


def test_cache_miss_on_distant_prompt(cache: InfinoSemanticCache) -> None:
    cache.update("the quick brown fox", "gpt-x", [Generation(text="answer")])
    assert cache.lookup("a completely unrelated question", "gpt-x") is None


def test_cache_clear(cache: InfinoSemanticCache) -> None:
    cache.update("remember me", "gpt-x", [Generation(text="ok")])
    assert cache.lookup("remember me", "gpt-x") is not None
    cache.clear()
    assert cache.lookup("remember me", "gpt-x") is None
