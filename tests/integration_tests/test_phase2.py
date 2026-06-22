"""Integration tests for phase-2: metadata filtering, hybrid, and MMR."""

import infino
import pyarrow as pa
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from langchain_infino import InfinoHybridRetriever, InfinoVectorStore

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
