"""Integration tests against a real Infino table on a temp directory.

Uses a deterministic fake embedding so the suite needs no model download,
and a tmp-dir (not ``memory://``) connection because ``delete`` requires
durable storage.
"""

import infino
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from langchain_infino import InfinoVectorStore

EMBED_DIM = 16
TEXTS = ["the cat sat on the mat", "a dog ran in the park", "quantum field theory"]


@pytest.fixture
def store(tmp_path) -> InfinoVectorStore:
    connection = infino.connect(str(tmp_path / "db"))
    return InfinoVectorStore.from_texts(
        TEXTS,
        DeterministicFakeEmbedding(size=EMBED_DIM),
        metadatas=[
            {"i": i, "kind": "animal" if i < 2 else "physics"} for i in range(3)
        ],
        connection=connection,
        table_name="docs",
        dim=EMBED_DIM,
    )


def test_similarity_search_returns_documents(store: InfinoVectorStore) -> None:
    docs = store.similarity_search(TEXTS[0], k=3)
    assert len(docs) == 3
    contents = {d.page_content for d in docs}
    assert TEXTS[0] in contents
    # The user id lands on Document.id; metadata round-trips on its own.
    assert all(d.id is not None for d in docs)
    assert any(d.metadata.get("kind") == "physics" for d in docs)


def test_similarity_search_with_score(store: InfinoVectorStore) -> None:
    results = store.similarity_search_with_score(TEXTS[0], k=2)
    assert len(results) == 2
    assert all(isinstance(score, float) for _, score in results)


def test_search_by_vector(store: InfinoVectorStore) -> None:
    embedding = DeterministicFakeEmbedding(size=EMBED_DIM).embed_query(TEXTS[2])
    docs = store.similarity_search_by_vector(embedding, k=1)
    assert len(docs) == 1


def test_add_texts_returns_ids_and_is_searchable(store: InfinoVectorStore) -> None:
    ids = store.add_texts(["a brand new sentence"], metadatas=[{"kind": "new"}])
    assert len(ids) == 1
    docs = store.similarity_search("a brand new sentence", k=4)
    assert any(d.page_content == "a brand new sentence" for d in docs)


def test_delete_removes_rows(store: InfinoVectorStore) -> None:
    ids = store.add_texts(["ephemeral row to delete"])
    assert store.delete(ids) is True
    docs = store.similarity_search("ephemeral row to delete", k=4)
    assert all(d.page_content != "ephemeral row to delete" for d in docs)


def test_delete_empty_is_noop(store: InfinoVectorStore) -> None:
    assert store.delete([]) is False
    assert store.delete(None) is False


def test_add_texts_rejects_mismatched_ids(store: InfinoVectorStore) -> None:
    with pytest.raises(ValueError, match="same length"):
        store.add_texts(["a", "b"], ids=["only-one"])


def test_add_texts_rejects_mismatched_metadatas(store: InfinoVectorStore) -> None:
    with pytest.raises(ValueError, match="same length"):
        store.add_texts(["a", "b"], metadatas=[{"k": 1}])


def test_relevance_scores_are_in_unit_interval(store: InfinoVectorStore) -> None:
    results = store.similarity_search_with_relevance_scores(TEXTS[0], k=3)
    assert results
    assert all(0.0 <= score <= 1.0 for _, score in results)
