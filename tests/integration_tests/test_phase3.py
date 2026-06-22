"""Integration tests for phase-3: self-query filter path and semantic cache."""

import infino
import pyarrow as pa
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.outputs import Generation
from langchain_core.structured_query import Comparator, Comparison, StructuredQuery

from langchain_infino import InfinoSemanticCache, InfinoTranslator, InfinoVectorStore

EMBED_DIM = 16
DOCS = [
    ("transformer attention is all you need", "ml", 2017),
    ("convolutional networks for images", "ml", 2015),
    ("general relativity field equations", "physics", 1915),
]


@pytest.fixture
def store(tmp_path) -> InfinoVectorStore:
    connection = infino.connect(str(tmp_path / "db"))
    return InfinoVectorStore.from_texts(
        [t for t, _, _ in DOCS],
        DeterministicFakeEmbedding(size=EMBED_DIM),
        metadatas=[{"category": c, "year": y} for _, c, y in DOCS],
        connection=connection,
        table_name="docs",
        dim=EMBED_DIM,
        metadata_columns=[
            pa.field("category", pa.large_utf8(), nullable=False),
            pa.field("year", pa.int64(), nullable=False),
        ],
    )


def test_translated_structured_query_filters(store: InfinoVectorStore) -> None:
    # Simulate what SelfQueryRetriever produces, without the LLM.
    structured = StructuredQuery(
        query="papers",
        filter=Comparison(comparator=Comparator.EQ, attribute="category", value="ml"),
        limit=None,
    )
    query, search_kwargs = InfinoTranslator().visit_structured_query(structured)
    docs = store.similarity_search(query, k=5, **search_kwargs)
    assert docs
    assert all(d.metadata["category"] == "ml" for d in docs)


def test_translated_range_query(store: InfinoVectorStore) -> None:
    structured = StructuredQuery(
        query="old work",
        filter=Comparison(comparator=Comparator.LT, attribute="year", value=2000),
        limit=None,
    )
    query, search_kwargs = InfinoTranslator().visit_structured_query(structured)
    docs = store.similarity_search(query, k=5, **search_kwargs)
    assert all(d.metadata["year"] < 2000 for d in docs)


# --- semantic cache ---


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
