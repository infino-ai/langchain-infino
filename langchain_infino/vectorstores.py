"""The :class:`InfinoVectorStore` LangChain vector store.

Phase 1: the core ``VectorStore`` contract over a single Infino table —
``add_texts``, ``from_texts``, vector similarity search (plain, scored, and
by-vector), and ``delete``. Metadata is round-tripped through a JSON
catch-all column; declared filterable metadata columns, hybrid (RRF)
retrieval, MMR, and the self-query translator are later phases.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Callable
from uuid import uuid4

import infino
import numpy as np
import pyarrow as pa
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores.utils import maximal_marginal_relevance

from langchain_infino._arrow import (
    METADATA_JSON_COLUMN,
    SCORE_COLUMN,
    rows_to_documents,
    serialize_metadata,
    sql_lit,
    vector_array,
)

if TYPE_CHECKING:
    from langchain_infino.retrievers import InfinoHybridRetriever

# Defaults chosen to match the engine's behavior and the LangChain idiom.
DEFAULT_K = 4
# The IVF builder clamps n_cent to <=64 below 100K rows, so 64 is the
# largest value that takes effect at small/medium scale without surprise.
DEFAULT_N_CENT = 64
DEFAULT_METRIC = "cosine"
DEFAULT_TEXT_COLUMN = "page_content"
DEFAULT_VECTOR_COLUMN = "embedding"
DEFAULT_ID_COLUMN = "doc_id"
DEFAULT_FETCH_K = 20
DEFAULT_LAMBDA_MULT = 0.5
# A structured filter is applied as a WHERE over the vector-search TVF, which
# ranks before filtering — so over-fetch candidates to refill the top-k after
# the predicate prunes. Approximate; a very selective filter may still
# under-return.
FILTER_OVERSAMPLE = 10

# LangChain's structured-filter operators → SQL comparison operators.
_FILTER_OPERATORS = {
    "$eq": "=",
    "$ne": "!=",
    "$gt": ">",
    "$gte": ">=",
    "$lt": "<",
    "$lte": "<=",
}

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
        metadata_columns: metadata keys to promote to real scalar columns so
            they can be filtered with the ``filter`` argument; any remaining
            metadata is kept in the JSON catch-all. Fixed at table creation —
            adding a filterable key later means recreating the table.
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
        metadata_columns: Sequence[pa.Field] = (),
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
        self._metadata_columns = list(metadata_columns)
        self._metadata_column_names = [f.name for f in self._metadata_columns]
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
        declared = set(self._metadata_column_names)

        # Column order must match the declared schema: id, text, vector,
        # *metadata_columns, _metadata_json.
        arrays: list[pa.Array] = [
            pa.array(ids, type=pa.large_utf8()),
            pa.array(texts, type=pa.large_utf8()),
            vector_array(vectors, self._dim),
        ]
        for field in self._metadata_columns:
            arrays.append(
                pa.array([m.get(field.name) for m in metadatas], type=field.type)
            )
        arrays.append(
            pa.array(
                [
                    serialize_metadata(
                        {k: v for k, v in m.items() if k not in declared}
                    )
                    for m in metadatas
                ],
                type=pa.large_utf8(),
            )
        )
        batch = pa.record_batch(arrays, schema=self._table.schema())
        self._table.append(batch)
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = DEFAULT_K,
        filter: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        embedding = self._embedding.embed_query(query)
        return self.similarity_search_by_vector(embedding, k, filter=filter, **kwargs)

    def similarity_search_by_vector(
        self,
        embedding: Sequence[float],
        k: int = DEFAULT_K,
        filter: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        return [doc for doc, _ in self._search(list(embedding), k, filter)]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = DEFAULT_K,
        filter: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        embedding = self._embedding.embed_query(query)
        return [
            (doc, score if score is not None else 0.0)
            for doc, score in self._search(embedding, k, filter)
        ]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = DEFAULT_K,
        fetch_k: int = DEFAULT_FETCH_K,
        lambda_mult: float = DEFAULT_LAMBDA_MULT,
        filter: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        # The vector column is not projectable and there is no point-lookup,
        # so re-embed the candidates' text to score them against each other.
        query_embedding = self._embedding.embed_query(query)
        candidates = self._search(query_embedding, fetch_k, filter)
        if not candidates:
            return []
        candidate_embeddings = self._embedding.embed_documents(
            [doc.page_content for doc, _ in candidates]
        )
        selected = maximal_marginal_relevance(
            np.array(query_embedding, dtype=np.float32),
            candidate_embeddings,
            k=k,
            lambda_mult=lambda_mult,
        )
        return [candidates[i][0] for i in selected]

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

    def _projection(self) -> list[str]:
        return [
            self._id_column,
            self._text_column,
            *self._metadata_column_names,
            METADATA_JSON_COLUMN,
            SCORE_COLUMN,
        ]

    def _search(
        self,
        embedding: Sequence[float],
        k: int,
        filter: Mapping[str, Any] | None = None,
    ) -> list[tuple[Document, float | None]]:
        projection = self._projection()
        if filter:
            where = _compile_filter(filter, self._metadata_column_names)
            # Over-fetch, filter, then trim — the TVF ranks before WHERE.
            columns = ", ".join(projection)
            sql = (
                f"SELECT {columns} FROM vector_search("
                f"{sql_lit(self._table_name)}, {sql_lit(self._vector_column)}, "
                f"{sql_lit(_vector_literal(embedding))}, {k * FILTER_OVERSAMPLE}) "
                f"WHERE {where} ORDER BY {SCORE_COLUMN} ASC LIMIT {k}"
            )
            result = self._connection.query_sql(sql)
        else:
            result = self._table.vector_search(
                self._vector_column, list(embedding), k, projection=projection
            )
        return rows_to_documents(
            result, id_column=self._id_column, text_column=self._text_column
        )

    def _hybrid_search(self, query: str, k: int = DEFAULT_K) -> list[Document]:
        """BM25 + vector retrieval fused by RRF in a single SQL call."""
        query_vector = _vector_literal(self._embedding.embed_query(query))
        sql = (
            f"SELECT * FROM hybrid_search("
            f"{sql_lit(self._table_name)}, {sql_lit(self._text_column)}, "
            f"{sql_lit(query)}, {sql_lit(self._vector_column)}, "
            f"{sql_lit(query_vector)}, {k}) ORDER BY {SCORE_COLUMN} DESC"
        )
        result = self._connection.query_sql(sql)
        return [
            doc
            for doc, _ in rows_to_documents(
                result, id_column=self._id_column, text_column=self._text_column
            )
        ]

    def as_hybrid_retriever(self, k: int = DEFAULT_K) -> InfinoHybridRetriever:
        """A retriever that fuses BM25 and vector search (RRF) per query."""
        from langchain_infino.retrievers import InfinoHybridRetriever

        return InfinoHybridRetriever(vectorstore=self, k=k)

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
        metadata_columns: Sequence[pa.Field] = (),
        **kwargs: Any,
    ) -> InfinoVectorStore:
        """Create the table, then embed and insert ``texts``."""
        schema = _build_schema(
            dim, text_column, vector_column, id_column, metadata_columns
        )
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
            metadata_columns=metadata_columns,
            table=table,
        )
        store.add_texts(texts, metadatas, ids=ids)
        return store


def _build_schema(
    dim: int,
    text_column: str,
    vector_column: str,
    id_column: str,
    metadata_columns: Sequence[pa.Field] = (),
) -> pa.Schema:
    """The declared table schema: id, text, embedding, *metadata, JSON."""
    return pa.schema(
        [
            pa.field(id_column, pa.large_utf8(), nullable=False),
            pa.field(text_column, pa.large_utf8(), nullable=False),
            pa.field(vector_column, pa.list_(pa.float32(), dim), nullable=False),
            *metadata_columns,
            pa.field(METADATA_JSON_COLUMN, pa.large_utf8(), nullable=False),
        ]
    )


def _vector_literal(embedding: Sequence[float]) -> str:
    """Render an embedding as the comma-separated string the SQL TVFs take."""
    return ",".join(str(float(x)) for x in embedding)


def _compile_filter(filter: Mapping[str, Any], allowed: Sequence[str]) -> str:
    """Compile a structured metadata filter into a SQL ``WHERE`` clause.

    Supports plain equality (``{"k": v}``) and the operator form
    (``{"k": {"$gt": 3}}``) over declared metadata columns. Filtering on a
    column that was not promoted out of the JSON catch-all is rejected — the
    engine cannot index into serialized JSON.
    """
    allowed_set = set(allowed)
    clauses: list[str] = []
    for key, condition in filter.items():
        if key not in allowed_set:
            raise ValueError(
                f"cannot filter on {key!r}: not a declared metadata column "
                f"(declared: {sorted(allowed_set)})"
            )
        if isinstance(condition, Mapping):
            for op, value in condition.items():
                if op == "$in":
                    items = ", ".join(_filter_literal(v) for v in value)
                    clauses.append(f"{key} IN ({items})")
                elif op in _FILTER_OPERATORS:
                    sql_op = _FILTER_OPERATORS[op]
                    clauses.append(f"{key} {sql_op} {_filter_literal(value)}")
                else:
                    raise ValueError(f"unsupported filter operator {op!r}")
        else:
            clauses.append(f"{key} = {_filter_literal(condition)}")
    return " AND ".join(clauses)


def _filter_literal(value: Any) -> str:
    """Render a filter value as a SQL literal."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return sql_lit(value)
    raise ValueError(f"unsupported filter value type: {type(value).__name__}")
