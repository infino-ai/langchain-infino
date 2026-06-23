"""LangChain's standard ``VectorStore`` compliance suite, run against Infino.

A fresh tmp-dir catalog per test gives each case an empty, isolated store.
The suite's default embedding is 6-dimensional; Infino's vector index requires
dim >= 16, so ``get_embeddings`` is overridden to a 16-dim deterministic fake.
"""

from collections.abc import Generator

import infino
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests.vectorstores import VectorStoreIntegrationTests

EMBED_DIM = 16


class TestInfinoVectorStore(VectorStoreIntegrationTests):
    @staticmethod
    def get_embeddings() -> Embeddings:
        return DeterministicFakeEmbedding(size=EMBED_DIM)

    @pytest.fixture
    def vectorstore(self, tmp_path) -> Generator[VectorStore, None, None]:
        from langchain_infino import InfinoVectorStore

        connection = infino.connect(str(tmp_path / "db"))
        yield InfinoVectorStore.from_texts(
            [],
            self.get_embeddings(),
            connection=connection,
            table_name="compliance",
            dim=EMBED_DIM,
        )

    # The published infino engine under-returns from vector_search after a
    # delete (no backfill past tombstones), so these delete cases fail at
    # k=1. Tracked upstream: infino-ai/infino#259. xfail (non-strict — it
    # only triggers on native x86, and passes on arm) until a fixed engine
    # release lands and the pin is bumped; then these revert to plain passes.
    _DELETE_BUG = pytest.mark.xfail(
        reason="infino#259: vector_search under-returns after delete", strict=False
    )

    @_DELETE_BUG
    def test_deleting_documents(self, vectorstore: VectorStore) -> None:
        super().test_deleting_documents(vectorstore)

    @_DELETE_BUG
    def test_deleting_bulk_documents(self, vectorstore: VectorStore) -> None:
        super().test_deleting_bulk_documents(vectorstore)

    @_DELETE_BUG
    async def test_deleting_documents_async(self, vectorstore: VectorStore) -> None:
        await super().test_deleting_documents_async(vectorstore)

    @_DELETE_BUG
    async def test_deleting_bulk_documents_async(
        self, vectorstore: VectorStore
    ) -> None:
        await super().test_deleting_bulk_documents_async(vectorstore)
