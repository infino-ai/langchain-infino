"""The store must hand every search option to the engine unchanged.

A dropped keyword is invisible end-to-end (the engine returns rows either
way), so these tests pin the forwarding itself. Vector serving carries no
tuning knobs — probe width and rerank budget are engine-decided — so the
forwarded surface is the filters and BM25 options plus the sentinel Nones.
"""

from typing import Any

import pyarrow as pa
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from langchain_infino.vectorstores import InfinoVectorStore

EMBED_DIM = 16

# The schema a store reads back to learn its embedding width and which
# metadata keys were promoted.
def _stub_schema(dim: int = EMBED_DIM) -> pa.Schema:
    return pa.schema(
        [
            pa.field("doc_id", pa.large_utf8(), nullable=False),
            pa.field("page_content", pa.large_utf8(), nullable=False),
            pa.field("embedding", pa.list_(pa.float32(), dim), nullable=False),
            pa.field("_metadata_json", pa.large_utf8(), nullable=False),
        ]
    )


class _RecordingTable:
    """Stand-in for ``infino.Table`` capturing the keywords it was called with."""

    def __init__(self) -> None:
        self.calls: dict[str, dict[str, Any]] = {}

    def _record(self, method: str, **kwargs: Any) -> pa.Table:
        self.calls[method] = kwargs
        return pa.table({})

    def schema(self) -> pa.Schema:
        return _stub_schema()

    def vector_search(
        self, column: str, query: Any, k: int, **kwargs: Any
    ) -> pa.Table:
        return self._record("vector_search", column=column, k=k, **kwargs)

    def bm25_search(self, column: str, query: str, k: int, **kwargs: Any) -> pa.Table:
        return self._record("bm25_search", column=column, query=query, k=k, **kwargs)

    def hybrid_search(
        self,
        text_column: str,
        text_query: str,
        vector_column: str,
        vector_query: Any,
        k: int,
        **kwargs: Any,
    ) -> pa.Table:
        return self._record("hybrid_search", text_query=text_query, k=k, **kwargs)


@pytest.fixture
def store() -> InfinoVectorStore:
    return InfinoVectorStore(
        connection=object(),  # type: ignore[arg-type]  # unused off the SQL path
        table_name="docs",
        embedding=DeterministicFakeEmbedding(size=EMBED_DIM),
        dim=EMBED_DIM,
        table=_RecordingTable(),  # type: ignore[arg-type]
    )


def _calls(store: InfinoVectorStore) -> dict[str, dict[str, Any]]:
    return store._table.calls  # type: ignore[union-attr]


def test_pushdown_prefilter_reaches_vector_search(store: InfinoVectorStore) -> None:
    store.similarity_search("q", k=3, filter_query="billing")
    call = _calls(store)["vector_search"]
    assert call["k"] == 3
    # The pushdown defaults to the indexed text column.
    assert call["filter_query"] == "billing"
    assert call["filter_column"] == "page_content"


def test_mmr_candidate_fetch_uses_fetch_k(store: InfinoVectorStore) -> None:
    store.max_marginal_relevance_search("q", k=2, fetch_k=9)
    call = _calls(store)["vector_search"]
    assert call["k"] == 9


def test_hybrid_search_forwards_k(store: InfinoVectorStore) -> None:
    store.as_hybrid_retriever(k=6).invoke("q")
    call = _calls(store)["hybrid_search"]
    assert call["k"] == 6


def test_bm25_stats_and_mode_reach_the_engine(store: InfinoVectorStore) -> None:
    store.as_bm25_retriever(k=4, mode="and", stats="global").invoke("q")
    call = _calls(store)["bm25_search"]
    assert call["k"] == 4
    assert call["mode"] == "and"
    assert call["stats"] == "global"


def test_options_default_to_none_so_the_engine_picks(store: InfinoVectorStore) -> None:
    # Omitted options must arrive as None, never as a client-side guess —
    # and the vector call must carry NO tuning kwargs at all.
    store.similarity_search("q", k=4)
    store.as_bm25_retriever(k=4).invoke("q")
    vector, bm25 = _calls(store)["vector_search"], _calls(store)["bm25_search"]
    assert "nprobe" not in vector and "rerank_mult" not in vector
    assert bm25["mode"] is None and bm25["stats"] is None
