"""The store must hand every search option to the engine unchanged.

The engine validates none of the tuning options — `nprobe=0` and
`nprobe=10**9` both return the same rows — so a dropped keyword is invisible
end-to-end. These tests pin the forwarding itself.
"""

from typing import Any

import pyarrow as pa
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from langchain_infino.vectorstores import InfinoVectorStore

EMBED_DIM = 16


class _RecordingTable:
    """Stand-in for ``infino.Table`` capturing the keywords it was called with."""

    def __init__(self) -> None:
        self.calls: dict[str, dict[str, Any]] = {}

    def _record(self, method: str, **kwargs: Any) -> pa.Table:
        self.calls[method] = kwargs
        return pa.table({})

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


def test_recall_knobs_reach_vector_search(store: InfinoVectorStore) -> None:
    store.similarity_search("q", k=7, nprobe=16, rerank_mult=4)
    call = _calls(store)["vector_search"]
    assert call["k"] == 7
    assert call["nprobe"] == 16
    assert call["rerank_mult"] == 4


def test_recall_knobs_reach_the_pushdown_prefilter(store: InfinoVectorStore) -> None:
    store.similarity_search("q", k=3, filter_query="billing", nprobe=8, rerank_mult=2)
    call = _calls(store)["vector_search"]
    assert call["nprobe"] == 8
    assert call["rerank_mult"] == 2
    # The pushdown defaults to the indexed text column.
    assert call["filter_query"] == "billing"
    assert call["filter_column"] == "page_content"


def test_recall_knobs_reach_mmr_candidate_fetch(store: InfinoVectorStore) -> None:
    store.max_marginal_relevance_search("q", k=2, fetch_k=9, nprobe=5, rerank_mult=3)
    call = _calls(store)["vector_search"]
    assert call["k"] == 9
    assert call["nprobe"] == 5
    assert call["rerank_mult"] == 3


def test_recall_knobs_reach_hybrid_search(store: InfinoVectorStore) -> None:
    store.as_hybrid_retriever(k=6, nprobe=12, rerank_mult=2).invoke("q")
    call = _calls(store)["hybrid_search"]
    assert call["k"] == 6
    assert call["nprobe"] == 12
    assert call["rerank_mult"] == 2


def test_bm25_stats_and_mode_reach_the_engine(store: InfinoVectorStore) -> None:
    store.as_bm25_retriever(k=4, mode="and", stats="global").invoke("q")
    call = _calls(store)["bm25_search"]
    assert call["k"] == 4
    assert call["mode"] == "and"
    assert call["stats"] == "global"


def test_options_default_to_none_so_the_engine_picks(store: InfinoVectorStore) -> None:
    # Omitted options must arrive as None, never as a client-side guess.
    store.similarity_search("q", k=4)
    store.as_bm25_retriever(k=4).invoke("q")
    vector, bm25 = _calls(store)["vector_search"], _calls(store)["bm25_search"]
    assert vector["nprobe"] is None and vector["rerank_mult"] is None
    assert bm25["mode"] is None and bm25["stats"] is None
