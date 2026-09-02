"""The engine capabilities the store exposes beyond ranked retrieval.

Counting, term matching, exact lookup and the maintenance calls all run in
the engine, so they are only meaningful against a real one.
"""

import infino
import pyarrow as pa
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from langchain_infino import GcReport, InfinoVectorStore

EMBED_DIM = 16
DOCS = [
    ("alpha vector search engine", "search"),
    ("beta lexical search engine", "search"),
    ("gamma hybrid fusion", "fusion"),
]


@pytest.fixture
def store(tmp_path) -> InfinoVectorStore:
    return InfinoVectorStore.from_texts(
        [text for text, _ in DOCS],
        DeterministicFakeEmbedding(size=EMBED_DIM),
        metadatas=[{"topic": topic} for _, topic in DOCS],
        connection=infino.connect(str(tmp_path / "db")),
        table_name="docs",
        dim=EMBED_DIM,
        metadata_columns=[pa.field("topic", pa.large_utf8(), nullable=False)],
    )


def test_count_matches_the_documents_that_match(store) -> None:
    assert store.count("search") == 2
    assert store.count("fusion") == 1
    assert store.count("nonexistentterm") == 0


def test_count_honours_the_boolean_mode(store) -> None:
    # "and" requires every term; only the two "search engine" docs have both.
    assert store.count("search engine", mode="and") == 2
    assert store.count("search fusion", mode="and") == 0
    assert store.count("search fusion", mode="or") == 3


def test_token_search_returns_whole_documents(store) -> None:
    docs = store.token_search("search")
    assert len(docs) == 2
    assert all(d.id is not None for d in docs)
    # Metadata survives the round trip, promoted column and all.
    assert {d.metadata["topic"] for d in docs} == {"search"}


def test_token_search_is_unranked_and_carries_no_score(store) -> None:
    # token_match is set membership, not relevance — there is no score column
    # to project, and asking for one would be an unknown-column error.
    docs = store.token_search("search")
    assert all("score" not in d.metadata for d in docs)


def test_token_search_honours_the_boolean_mode(store) -> None:
    assert len(store.token_search("search fusion", mode="and")) == 0
    assert len(store.token_search("search fusion", mode="or")) == 3


def test_exact_search_resolves_a_verbatim_key(store) -> None:
    ids = store.add_texts(["delta added later"], metadatas=[{"topic": "extra"}])
    found = store.exact_search(ids[0], "doc_id")
    assert [d.id for d in found] == [ids[0]]
    assert found[0].page_content == "delta added later"


def test_exact_search_on_a_missing_value_is_empty(store) -> None:
    assert store.exact_search("no-such-id", "doc_id") == []


def test_schema_reports_the_declared_columns(store) -> None:
    names = [field.name for field in store.schema()]
    assert names == [
        "doc_id",
        "page_content",
        "embedding",
        "topic",
        "_metadata_json",
    ]


def test_accessors_expose_the_underlying_handles(store) -> None:
    assert store.table_name == "docs"
    assert store.metric == "cosine"
    # The escape hatch for engine calls the store does not wrap.
    assert store.connection.list_tables() == ["docs"]
    assert store.table.count("page_content", "search") == 2


def test_optimize_runs_and_leaves_the_table_searchable(store) -> None:
    store.optimize()
    assert len(store.similarity_search("search", k=2)) == 2


def test_optimize_accepts_engine_tuning_bounds(store) -> None:
    store.optimize(max_memory_mb=64, min_fill_percent=50)
    assert len(store.similarity_search("search", k=2)) == 2


def test_gc_reclaims_after_optimize(store) -> None:
    # Compaction leaves the pre-merge superfiles unreferenced; a zero grace
    # period spares nothing, so they are reclaimed immediately.
    store.optimize()
    report = store.gc(0.0)
    assert isinstance(report, GcReport)
    assert report.objects_deleted >= 1
    assert report.delete_errors == 0
    # The live snapshot is untouched.
    assert len(store.similarity_search("search", k=2)) == 2


def test_gc_grace_period_spares_recent_objects(store) -> None:
    store.optimize()
    # Everything here was written seconds ago, so a wide grace spares it all.
    report = store.gc(3600.0)
    assert report.objects_deleted == 0
    assert report.objects_skipped_too_new >= 1


def test_connect_creates_and_opens_a_store_from_a_uri(tmp_path) -> None:
    embedding = DeterministicFakeEmbedding(size=EMBED_DIM)
    uri = str(tmp_path / "from-uri")

    created = InfinoVectorStore.connect(
        uri, embedding, "docs", dim=EMBED_DIM, create=True
    )
    ids = created.add_texts(["alpha vector search"])

    # A second connection to the same URI sees the committed table.
    reopened = InfinoVectorStore.connect(uri, embedding, "docs", dim=EMBED_DIM)
    assert [d.id for d in reopened.get_by_ids(ids)] == ids


def test_connect_forwards_cache_configuration_to_a_working_store(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    store = InfinoVectorStore.connect(
        str(tmp_path / "db"),
        DeterministicFakeEmbedding(size=EMBED_DIM),
        "docs",
        dim=EMBED_DIM,
        create=True,
        cache_dir=str(cache_dir),
        cache_budget_bytes=64 << 20,
        connection_memory_budget_bytes=512 << 20,
        validate=True,
    )
    store.add_texts(["alpha vector search"])
    assert len(store.similarity_search("search", k=1)) == 1
    assert cache_dir.is_dir()


def test_open_or_create_creates_then_reattaches(tmp_path) -> None:
    embedding = DeterministicFakeEmbedding(size=EMBED_DIM)
    uri = str(tmp_path / "db")

    created = InfinoVectorStore.open_or_create(
        infino.connect(uri), "docs", embedding, dim=EMBED_DIM
    )
    ids = created.add_texts(["alpha vector search"])

    # Same call again, on a separate connection: reattaches, keeps the data.
    reopened = InfinoVectorStore.open_or_create(
        infino.connect(uri), "docs", embedding, dim=EMBED_DIM
    )
    assert [d.id for d in reopened.get_by_ids(ids)] == ids


def test_open_or_create_yields_a_writable_empty_table(tmp_path) -> None:
    # A table created but never written is openable and usable, so a store
    # need not be populated before another process attaches to it.
    embedding = DeterministicFakeEmbedding(size=EMBED_DIM)
    uri = str(tmp_path / "db")

    InfinoVectorStore.open_or_create(
        infino.connect(uri), "docs", embedding, dim=EMBED_DIM
    )
    later = InfinoVectorStore.open_or_create(
        infino.connect(uri), "docs", embedding, dim=EMBED_DIM
    )
    assert later.similarity_search("anything", k=2) == []
    later.add_texts(["alpha vector search"])
    assert len(later.similarity_search("search", k=2)) == 1


def test_open_or_create_rejects_an_unbuildable_table(tmp_path) -> None:
    # The race fallback must not swallow a create the engine refused outright.
    with pytest.raises(ValueError):
        InfinoVectorStore.open_or_create(
            infino.connect(str(tmp_path / "db")),
            "docs",
            DeterministicFakeEmbedding(size=4),
            dim=4,  # below the engine's supported minimum of 16
        )


def test_dim_is_inferred_from_the_embedding_when_creating(tmp_path) -> None:
    store = InfinoVectorStore.from_texts(
        ["alpha vector search"],
        DeterministicFakeEmbedding(size=EMBED_DIM),
        connection=infino.connect(str(tmp_path / "db")),
    )
    assert store.schema().field("embedding").type.list_size == EMBED_DIM
    assert len(store.similarity_search("search", k=1)) == 1


def test_scalar_metadata_is_filterable_without_declaring_columns(tmp_path) -> None:
    store = InfinoVectorStore.from_texts(
        ["alpha search", "beta search", "gamma search"],
        DeterministicFakeEmbedding(size=EMBED_DIM),
        metadatas=[
            {"src": "a", "year": 2024, "ok": True},
            {"src": "b", "year": 2023, "ok": False},
            {"src": "c"},
        ],
        connection=infino.connect(str(tmp_path / "db")),
    )
    for filter_, expected in [
        ({"src": "a"}, ["a"]),
        ({"year": {"$gte": 2024}}, ["a"]),
        ({"ok": True}, ["a"]),
        ({"year": {"$lt": 2024}}, ["b"]),
    ]:
        found = store.similarity_search("search", k=3, filter=filter_)
        assert [d.metadata["src"] for d in found] == expected


def test_a_document_missing_a_promoted_key_gains_no_null(tmp_path) -> None:
    store = InfinoVectorStore.from_texts(
        ["alpha search", "beta search"],
        DeterministicFakeEmbedding(size=EMBED_DIM),
        metadatas=[{"src": "a", "year": 2024}, {"src": "b"}],
        connection=infino.connect(str(tmp_path / "db")),
    )
    sparse = store.similarity_search("search", k=2, filter={"src": "b"})
    assert sparse[0].metadata == {"src": "b"}


def test_metadata_the_engine_cannot_index_still_round_trips(tmp_path) -> None:
    # Nested values and names the engine reserves stay in the JSON catch-all:
    # not filterable, but not lost either.
    store = InfinoVectorStore.from_texts(
        ["alpha search"],
        DeterministicFakeEmbedding(size=EMBED_DIM),
        metadatas=[{"tags": ["x", "y"], "score": 1.5, "src": "a"}],
        connection=infino.connect(str(tmp_path / "db")),
    )
    assert [f.name for f in store.metadata_columns] == ["src"]
    found = store.similarity_search("search", k=1)
    assert found[0].metadata == {"tags": ["x", "y"], "score": 1.5, "src": "a"}


def test_reopening_needs_neither_dim_nor_columns(tmp_path) -> None:
    embedding = DeterministicFakeEmbedding(size=EMBED_DIM)
    uri = str(tmp_path / "db")
    InfinoVectorStore.from_texts(
        ["alpha search", "beta search"],
        embedding,
        metadatas=[{"src": "a"}, {"src": "b"}],
        connection=infino.connect(uri),
        table_name="docs",
    )
    # Both are declared by the table, so neither has to be repeated.
    reopened = InfinoVectorStore.connect(uri, embedding, "docs")
    found = reopened.similarity_search("search", k=2, filter={"src": "b"})
    assert [d.metadata["src"] for d in found] == ["b"]


def test_declaring_a_reserved_column_name_fails_at_creation(tmp_path) -> None:
    # Better here than as a duplicate-column error on the first filtered query.
    with pytest.raises(ValueError, match="reserves"):
        InfinoVectorStore.from_texts(
            ["alpha search"],
            DeterministicFakeEmbedding(size=EMBED_DIM),
            connection=infino.connect(str(tmp_path / "db")),
            metadata_columns=[pa.field("score", pa.float64(), nullable=True)],
        )


def test_delete_with_no_ids_empties_the_table(store) -> None:
    # The VectorStore contract: `ids=None` means delete everything.
    assert store.delete() is True
    assert store.similarity_search("search", k=3) == []


def test_delete_with_an_empty_list_deletes_nothing(store) -> None:
    # An empty list is not the same as None.
    assert store.delete([]) is False
    assert len(store.similarity_search("search", k=3)) == len(DOCS)


def test_delete_by_filter_removes_only_the_matches(store) -> None:
    assert store.delete(filter={"topic": "fusion"}) is True
    remaining = store.token_search("search")
    assert {d.metadata["topic"] for d in remaining} == {"search"}


def test_delete_rejects_ids_and_filter_together(store) -> None:
    with pytest.raises(ValueError, match="not both"):
        store.delete(["x"], filter={"topic": "fusion"})


def test_delete_by_predicate_reports_what_it_removed(store) -> None:
    stats = store.delete_by_predicate("topic = 'search'")
    assert stats.matched == 2
    assert stats.n_tombstoned == 2
    assert len(store.token_search("search")) == 0


def test_deleting_absent_ids_still_reports_success(store) -> None:
    # Idempotent: nothing matched, but the delete did not fail.
    assert store.delete(["no-such-id"]) is True


def test_drop_removes_the_table(store) -> None:
    assert store.connection.list_tables() == ["docs"]
    store.drop()
    assert store.connection.list_tables() == []


def test_drop_without_purge_keeps_the_stored_objects(store) -> None:
    store.drop(purge=False)
    assert store.connection.list_tables() == []
