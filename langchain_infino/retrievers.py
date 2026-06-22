"""Retrievers exposing Infino's multi-modal retrieval to LangChain.

The hybrid retriever fuses BM25 and vector search by reciprocal-rank fusion
in a single SQL call — no separate reranking round-trip.
"""

from __future__ import annotations

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from langchain_infino.vectorstores import DEFAULT_K, InfinoVectorStore


class InfinoHybridRetriever(BaseRetriever):
    """Retriever that fuses BM25 and vector search (RRF) per query.

    Wraps an :class:`InfinoVectorStore`; the fusion runs entirely in the
    engine via the ``hybrid_search`` SQL function.
    """

    vectorstore: InfinoVectorStore
    k: int = DEFAULT_K

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self.vectorstore._hybrid_search(query, self.k)
