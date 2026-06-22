"""Integration tests for the self-query filter path (translator -> SQL WHERE)."""

import infino
import pyarrow as pa
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.structured_query import Comparator, Comparison, StructuredQuery

from langchain_infino import InfinoTranslator, InfinoVectorStore

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
