"""LangChain's standard compliance suites, run against Infino.

This package ships three LangChain component types — a vector store, a cache
and two retrievers — so all three standard suites run here.

A fresh tmp-dir catalog per test gives each case an empty, isolated store. The
suites' default embedding is 6-dimensional; Infino's vector index requires
dim >= 16, so ``get_embeddings`` is overridden to a 16-dim deterministic fake.
"""

from collections.abc import Generator

import infino
import pytest
from langchain_core.caches import BaseCache
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests import (
    AsyncCacheTestSuite,
    RetrieversIntegrationTests,
    SyncCacheTestSuite,
)
from langchain_tests.integration_tests.vectorstores import VectorStoreIntegrationTests

from langchain_infino import (
    InfinoBM25Retriever,
    InfinoHybridRetriever,
    InfinoSemanticCache,
    InfinoVectorStore,
)

EMBED_DIM = 16


class TestInfinoVectorStore(VectorStoreIntegrationTests):
    @staticmethod
    def get_embeddings() -> Embeddings:
        return DeterministicFakeEmbedding(size=EMBED_DIM)

    @pytest.fixture
    def vectorstore(self, tmp_path) -> Generator[VectorStore, None, None]:
        connection = infino.connect(str(tmp_path / "db"))
        yield InfinoVectorStore.from_texts(
            [],
            self.get_embeddings(),
            connection=connection,
            table_name="compliance",
            dim=EMBED_DIM,
        )


class TestInfinoSemanticCacheSync(SyncCacheTestSuite):
    @pytest.fixture
    def cache(self, tmp_path) -> BaseCache:
        return InfinoSemanticCache(
            infino.connect(str(tmp_path / "cache")),
            self.get_embeddings(),
            dim=EMBED_DIM,
        )

    @staticmethod
    def get_embeddings() -> Embeddings:
        return DeterministicFakeEmbedding(size=EMBED_DIM)


class TestInfinoSemanticCacheAsync(AsyncCacheTestSuite):
    @pytest.fixture
    def cache(self, tmp_path) -> BaseCache:
        return InfinoSemanticCache(
            infino.connect(str(tmp_path / "cache")),
            self.get_embeddings(),
            dim=EMBED_DIM,
        )

    @staticmethod
    def get_embeddings() -> Embeddings:
        return DeterministicFakeEmbedding(size=EMBED_DIM)


# The retriever suite asks for three results from one query, and BM25 can only
# return documents that actually match — so every document here carries the
# query term.
RETRIEVER_CORPUS = [
    "alpha vector search",
    "beta lexical search",
    "gamma hybrid search",
    "delta federated search",
]
RETRIEVER_QUERY = "search"


class _InfinoRetrieverTests(RetrieversIntegrationTests):
    """Shared wiring: a populated store, reachable from the suite's properties.

    The suite reads its parameters from properties rather than fixtures, so the
    store is built by an autouse fixture and stashed on the instance.
    """

    @pytest.fixture(autouse=True)
    def _corpus(self, tmp_path) -> None:
        self._store = InfinoVectorStore.from_texts(
            RETRIEVER_CORPUS,
            DeterministicFakeEmbedding(size=EMBED_DIM),
            connection=infino.connect(str(tmp_path / "db")),
            table_name="docs",
            dim=EMBED_DIM,
        )

    @property
    def retriever_constructor_params(self) -> dict:
        return {"vectorstore": self._store}

    @property
    def retriever_query_example(self) -> str:
        return RETRIEVER_QUERY


class TestInfinoBM25Retriever(_InfinoRetrieverTests):
    @property
    def retriever_constructor(self) -> type[InfinoBM25Retriever]:
        return InfinoBM25Retriever


class TestInfinoHybridRetriever(_InfinoRetrieverTests):
    @property
    def retriever_constructor(self) -> type[InfinoHybridRetriever]:
        return InfinoHybridRetriever
