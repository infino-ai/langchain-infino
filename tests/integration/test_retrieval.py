"""Integration tests for filtering, hybrid/BM25 retrieval, MMR, and SQL."""

import infino
import pyarrow as pa
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from langchain_infino import (
    InfinoBM25Retriever,
    InfinoHybridRetriever,
    InfinoVectorStore,
)

EMBED_DIM = 16
DOCS = [
    ("the cat sat on the mat", "animal", 2021),
    ("a dog ran in the park", "animal", 2022),
    ("quantum field theory lecture", "physics", 2023),
    ("training a neural network", "ml", 2024),
    ("gradient descent for deep learning", "ml", 2024),
]


@pytest.fixture
def store(tmp_path) -> InfinoVectorStore:
    connection = infino.connect(str(tmp_path / "db"))
    return InfinoVectorStore.from_texts(
        [text for text, _, _ in DOCS],
        DeterministicFakeEmbedding(size=EMBED_DIM),
        metadatas=[{"category": c, "year": y, "note": "x"} for _, c, y in DOCS],
        connection=connection,
        table_name="docs",
        dim=EMBED_DIM,
        metadata_columns=[
            pa.field("category", pa.large_utf8(), nullable=False),
            pa.field("year", pa.int64(), nullable=False),
        ],
    )


def test_declared_columns_round_trip_into_metadata(store: InfinoVectorStore) -> None:
    docs = store.similarity_search(DOCS[0][0], k=5)
    doc = next(d for d in docs if d.page_content == DOCS[0][0])
    assert doc.metadata["category"] == "animal"
    assert doc.metadata["year"] == 2021
    # Undeclared key stays in the JSON catch-all.
    assert doc.metadata["note"] == "x"


def test_equality_filter_restricts_results(store: InfinoVectorStore) -> None:
    docs = store.similarity_search("anything", k=5, filter={"category": "ml"})
    assert docs
    assert all(d.metadata["category"] == "ml" for d in docs)


def test_operator_filter(store: InfinoVectorStore) -> None:
    docs = store.similarity_search("anything", k=5, filter={"year": {"$gte": 2023}})
    assert docs
    assert all(d.metadata["year"] >= 2023 for d in docs)


def test_filter_on_undeclared_column_raises(store: InfinoVectorStore) -> None:
    with pytest.raises(ValueError, match="not a declared metadata column"):
        store.similarity_search("x", k=3, filter={"note": "x"})


def test_hybrid_retriever(store: InfinoVectorStore) -> None:
    retriever = store.as_hybrid_retriever(k=3)
    assert isinstance(retriever, InfinoHybridRetriever)
    docs = retriever.invoke("neural network")
    assert docs
    assert len(docs) <= 3
    assert all(isinstance(d.page_content, str) for d in docs)


def test_mmr_returns_k_documents(store: InfinoVectorStore) -> None:
    docs = store.max_marginal_relevance_search("learning", k=3, fetch_k=5)
    assert len(docs) == 3
    # No duplicates among the selected documents.
    assert len({d.page_content for d in docs}) == 3


def test_mmr_with_filter(store: InfinoVectorStore) -> None:
    docs = store.max_marginal_relevance_search(
        "learning", k=2, fetch_k=5, filter={"category": "ml"}
    )
    assert all(d.metadata["category"] == "ml" for d in docs)


# --- text-pushdown pre-filter (vector kNN restricted to FTS matches) ---


def test_pushdown_restricts_to_fts_matches(store: InfinoVectorStore) -> None:
    # Only the two docs containing "network"/"neural" can come back.
    docs = store.similarity_search("anything", k=5, filter_query="neural")
    assert docs
    assert all("neural" in d.page_content for d in docs)


def test_pushdown_and_mode_requires_all_terms(store: InfinoVectorStore) -> None:
    docs = store.similarity_search(
        "anything", k=5, filter_query="deep learning", filter_mode="and"
    )
    assert docs
    assert all("deep" in d.page_content and "learning" in d.page_content for d in docs)


def test_pushdown_no_match_returns_empty(store: InfinoVectorStore) -> None:
    assert store.similarity_search("anything", k=5, filter_query="zebra") == []


def test_pushdown_via_retriever_search_kwargs(store: InfinoVectorStore) -> None:
    retriever = store.as_retriever(search_kwargs={"k": 5, "filter_query": "dog"})
    docs = retriever.invoke("anything")
    assert docs
    assert all("dog" in d.page_content for d in docs)


def test_pushdown_with_mmr(store: InfinoVectorStore) -> None:
    docs = store.max_marginal_relevance_search(
        "anything", k=2, fetch_k=5, filter_query="learning"
    )
    assert docs
    assert all("learning" in d.page_content for d in docs)


def test_structured_filter_and_pushdown_together_raises(
    store: InfinoVectorStore,
) -> None:
    with pytest.raises(ValueError, match="not both"):
        store.similarity_search(
            "x", k=3, filter={"category": "ml"}, filter_query="neural"
        )


# --- lexical BM25 retrieval ---


def test_bm25_retriever(store: InfinoVectorStore) -> None:
    retriever = store.as_bm25_retriever(k=3)
    assert isinstance(retriever, InfinoBM25Retriever)
    docs = retriever.invoke("neural network")
    assert docs
    assert any("neural" in d.page_content for d in docs)


def test_bm25_and_mode_requires_all_terms(store: InfinoVectorStore) -> None:
    retriever = store.as_bm25_retriever(k=5, mode="and")
    docs = retriever.invoke("deep learning")
    assert docs
    assert all("deep" in d.page_content and "learning" in d.page_content for d in docs)


# --- SQL-native escape hatch ---


def test_search_by_sql_maps_rows_to_documents(store: InfinoVectorStore) -> None:
    docs = store.search_by_sql(
        "SELECT doc_id, page_content, category, year, _metadata_json "
        "FROM docs WHERE category = 'ml'"
    )
    assert docs
    assert all(d.metadata["category"] == "ml" for d in docs)
    assert all(d.id is not None for d in docs)
