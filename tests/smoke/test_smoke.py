"""End-to-end smoke test of the public API against a real engine.

Runs against the *installed* package (e.g. a freshly built wheel) to catch
packaging and published-dependency regressions — missing package data, wrong
pins, an API drift versus the published ``infino``. Depth of behavior is the
unit and integration suites' job; this just confirms the wiring holds.
"""

import infino
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.outputs import Generation

import langchain_infino
from langchain_infino import (
    InfinoBM25Retriever,
    InfinoHybridRetriever,
    InfinoSemanticCache,
    InfinoTranslator,
    InfinoVectorStore,
)

DIM = 16
TEXTS = ["alpha vector search", "beta lexical search", "gamma hybrid fusion"]


def test_public_api_smoke(tmp_path) -> None:
    assert langchain_infino.__version__  # version metadata resolves

    connection = infino.connect(str(tmp_path / "db"))
    embedding = DeterministicFakeEmbedding(size=DIM)
    store = InfinoVectorStore.from_texts(
        TEXTS, embedding, connection=connection, table_name="smoke", dim=DIM
    )

    ids = store.add_texts(["delta added later"])
    assert len(ids) == 1
    assert store.similarity_search("search", k=2)
    assert store.get_by_ids(ids)[0].id == ids[0]

    hybrid = store.as_hybrid_retriever(k=2)
    assert isinstance(hybrid, InfinoHybridRetriever)
    assert hybrid.invoke("search")

    bm25 = store.as_bm25_retriever(k=2)
    assert isinstance(bm25, InfinoBM25Retriever)
    assert bm25.invoke("search")

    assert store.search_by_sql(
        "SELECT doc_id, page_content, _metadata_json FROM smoke"
    )

    cache = InfinoSemanticCache(
        connection, embedding, dim=DIM, table_name="smoke_cache"
    )
    cache.update("ping", "model-x", [Generation(text="pong")])
    assert cache.lookup("ping", "model-x") is not None

    assert InfinoTranslator().allowed_comparators
