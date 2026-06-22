"""The :class:`InfinoVectorStore` LangChain vector store.

Phase 1: the core ``VectorStore`` contract over a single Infino table —
``add_texts``, ``from_texts``, vector similarity search (plain, scored, and
by-vector), and ``delete``. Metadata is round-tripped through a JSON
catch-all column; declared filterable metadata columns, hybrid (RRF)
retrieval, MMR, and the self-query translator are later phases.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Callable
from uuid import uuid4

import infino
import pyarrow as pa
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_infino._arrow import (
    METADATA_JSON_COLUMN,
    SCORE_COLUMN,
    rows_to_documents,
    serialize_metadata,
    sql_lit,
    vector_array,
)

# Defaults chosen to match the engine's behavior and the LangChain idiom.
DEFAULT_K = 4
# The IVF builder clamps n_cent to <=64 below 100K rows, so 64 is the
# largest value that takes effect at small/medium scale without surprise.
DEFAULT_N_CENT = 64
DEFAULT_METRIC = "cosine"
DEFAULT_TEXT_COLUMN = "page_content"
DEFAULT_VECTOR_COLUMN = "embedding"
DEFAULT_ID_COLUMN = "doc_id"

# Per-metric maps from Infino's raw distance/score to a [0, 1] relevance
# where higher is more relevant, for similarity_search_with_relevance_scores.
_RELEVANCE_FNS: dict[str, Callable[[float], float]] = {
    # Cosine distance is 1 - cosine_similarity, already in [0, 2]; clamp.
    "cosine": lambda d: max(0.0, min(1.0, 1.0 - d)),
    # Squared-L2 is unbounded above; map monotonically into (0, 1].
    "l2sq": lambda d: 1.0 / (1.0 + d),
    "l2": lambda d: 1.0 / (1.0 + d),
}


class InfinoVectorStore(VectorStore):
    """LangChain ``VectorStore`` backed by a single Infino table.

    The table is created with three columns — the document id, the text, and
    the embedding — plus a JSON catch-all that round-trips arbitrary
    metadata. The id and text columns are FTS-indexed: the id so ``delete``
    can prune by ``exact_match``, the text so later phases can add BM25 and
    hybrid retrieval over the same data.

    Args:
        connection: a live :class:`infino.Connection`.
        table_name: the table to open (must already exist; use
            :meth:`from_texts` to create and populate one).
        embedding: the LangChain embeddings to use for query and documents.
        dim: embedding dimension; must match the table's vector column and
            lie in the engine's supported range [16, 4096].
        metric: distance metric — ``"cosine"`` (default), ``"l2sq"`` /
            ``"l2"``, or ``"negdot"`` / ``"dot"``.
        n_cent: IVF centroid count; size to the table's scale.
        text_column / vector_column / id_column: column names.
    """

    def __init__(
        self,
        connection: infino.Connection,
        table_name: str,
        embedding: Embeddings,
        *,
        dim: int,
        metric: str = DEFAULT_METRIC,
        n_cent: int = DEFAULT_N_CENT,
        text_column: str = DEFAULT_TEXT_COLUMN,
        vector_column: str = DEFAULT_VECTOR_COLUMN,
        id_column: str = DEFAULT_ID_COLUMN,
        table: infino.Table | None = None,
    ) -> None:
        self._connection = connection
        self._table_name = table_name
        self._embedding = embedding
        self._dim = dim
        self._metric = metric
        self._n_cent = n_cent
        self._text_column = text_column
        self._vector_column = vector_column
        self._id_column = id_column
        # A table's manifest is written on its first commit, so open_table
        # only succeeds once it holds data. from_texts hands in the handle
        # returned by create_table; opening by name serves already-populated
        # tables.
        self._table = table if table is not None else connection.open_table(table_name)

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        texts = list(texts)
        if not texts:
            return []

        if ids is None:
            ids = [uuid4().hex for _ in texts]
        elif len(ids) != len(texts):
            raise ValueError("ids and texts must have the same length")

        if metadatas is None:
            metadatas = [{} for _ in texts]
        elif len(metadatas) != len(texts):
            raise ValueError("metadatas and texts must have the same length")

        vectors = self._embedding.embed_documents(texts)
        batch = pa.record_batch(
            [
                pa.array(ids, type=pa.large_utf8()),
                pa.array(texts, type=pa.large_utf8()),
                vector_array(vectors, self._dim),
                pa.array(
                    [serialize_metadata(m) for m in metadatas],
                    type=pa.large_utf8(),
                ),
            ],
            schema=self._table.schema(),
        )
        self._table.append(batch)
        return ids

    def similarity_search(
        self, query: str, k: int = DEFAULT_K, **kwargs: Any
    ) -> list[Document]:
        embedding = self._embedding.embed_query(query)
        return self.similarity_search_by_vector(embedding, k, **kwargs)

    def similarity_search_by_vector(
        self, embedding: Sequence[float], k: int = DEFAULT_K, **kwargs: Any
    ) -> list[Document]:
        return [doc for doc, _ in self._search(list(embedding), k)]

    def similarity_search_with_score(
        self, query: str, k: int = DEFAULT_K, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        embedding = self._embedding.embed_query(query)
        return [
            (doc, score if score is not None else 0.0)
            for doc, score in self._search(embedding, k)
        ]

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        if not ids:
            return False
        id_list = ", ".join(sql_lit(i) for i in ids)
        self._table.delete(f"{self._id_column} IN ({id_list})")
        return True

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        try:
            return _RELEVANCE_FNS[self._metric]
        except KeyError:
            raise ValueError(
                f"no relevance-score normalization for metric {self._metric!r}; "
                f"use similarity_search_with_score for the raw distance"
            ) from None

    def _search(
        self, embedding: Sequence[float], k: int
    ) -> list[tuple[Document, float | None]]:
        result = self._table.vector_search(
            self._vector_column,
            list(embedding),
            k,
            projection=[
                self._id_column,
                self._text_column,
                METADATA_JSON_COLUMN,
                SCORE_COLUMN,
            ],
        )
        return rows_to_documents(
            result, id_column=self._id_column, text_column=self._text_column
        )

    @classmethod
    def from_texts(  # type: ignore[override]  # requires engine params (connection, table_name, dim) the base signature lacks
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        *,
        connection: infino.Connection,
        table_name: str,
        dim: int,
        ids: list[str] | None = None,
        metric: str = DEFAULT_METRIC,
        n_cent: int = DEFAULT_N_CENT,
        text_column: str = DEFAULT_TEXT_COLUMN,
        vector_column: str = DEFAULT_VECTOR_COLUMN,
        id_column: str = DEFAULT_ID_COLUMN,
        **kwargs: Any,
    ) -> InfinoVectorStore:
        """Create the table, then embed and insert ``texts``."""
        schema = _build_schema(dim, text_column, vector_column, id_column)
        indexes = (
            infino.IndexSpec()
            .fts(text_column)
            .fts(id_column)
            .vector(vector_column, dim, n_cent, metric)
        )
        table = connection.create_table(table_name, schema, indexes)

        store = cls(
            connection,
            table_name,
            embedding,
            dim=dim,
            metric=metric,
            n_cent=n_cent,
            text_column=text_column,
            vector_column=vector_column,
            id_column=id_column,
            table=table,
        )
        store.add_texts(texts, metadatas, ids=ids)
        return store


def _build_schema(
    dim: int, text_column: str, vector_column: str, id_column: str
) -> pa.Schema:
    """The declared table schema: id, text, embedding, JSON metadata."""
    return pa.schema(
        [
            pa.field(id_column, pa.large_utf8(), nullable=False),
            pa.field(text_column, pa.large_utf8(), nullable=False),
            pa.field(vector_column, pa.list_(pa.float32(), dim), nullable=False),
            pa.field(METADATA_JSON_COLUMN, pa.large_utf8(), nullable=False),
        ]
    )
