"""Retrievers exposing Infino's multi-modal retrieval to LangChain.

Hybrid fuses BM25 and vector search by reciprocal-rank fusion in one SQL
call; BM25 is lexical-only. Both wrap an :class:`InfinoVectorStore` and run
entirely in the engine.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from langchain_infino.vectorstores import DEFAULT_K, InfinoVectorStore, SearchMode


class InfinoHybridRetriever(BaseRetriever):
    """Retriever that fuses BM25 and vector search (RRF) per query.

    The fusion runs entirely in the engine via ``hybrid_search`` — no
    separate reranking round-trip.
    """

    vectorstore: InfinoVectorStore
    k: int = DEFAULT_K

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self.vectorstore._hybrid_search(query, self.k)


class InfinoBM25Retriever(BaseRetriever):
    """Lexical BM25 retriever over the FTS-indexed text column.

    ``mode`` joins query terms: ``"or"`` (default) matches any, ``"and"``
    requires all.
    """

    vectorstore: InfinoVectorStore
    k: int = DEFAULT_K
    mode: Optional[SearchMode] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self.vectorstore._bm25_search(query, self.k, self.mode)
