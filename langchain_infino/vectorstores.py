"""The :class:`InfinoVectorStore` LangChain vector store.

Maps the ``VectorStore`` contract onto one Infino table: text and embedding
in dedicated columns, the doc id in an FTS-indexed column, metadata either
promoted to scalar columns (filterable) or kept in a JSON catch-all. Vector,
filtered, MMR, and hybrid (RRF) retrieval all run over that one table.
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
    from langchain_infino.retrievers import InfinoBM25Retriever, InfinoHybridRetriever

DEFAULT_K = 4
# IVF builder clamps n_cent to <=64 below 100K rows; 64 is the effective max.
DEFAULT_N_CENT = 64
DEFAULT_METRIC = "cosine"
DEFAULT_TEXT_COLUMN = "page_content"
DEFAULT_VECTOR_COLUMN = "embedding"
DEFAULT_ID_COLUMN = "doc_id"
DEFAULT_FETCH_K = 20
DEFAULT_LAMBDA_MULT = 0.5
# Structured filter is a WHERE applied after the TVF ranks, so over-fetch to
# refill the top-k. A very selective filter may still under-return.
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

# Logical operators that join sub-filters; "$not" is handled separately.
_LOGICAL_OPERATORS = {"$and": " AND ", "$or": " OR "}

# Map Infino's raw distance to a [0, 1] relevance, higher = more relevant.
_RELEVANCE_FNS: dict[str, Callable[[float], float]] = {
    # Cosine distance is 1 - cosine_similarity, already in [0, 2]; clamp.
    "cosine": lambda d: max(0.0, min(1.0, 1.0 - d)),
    # Squared-L2 is unbounded above; map monotonically into (0, 1].
    "l2sq": lambda d: 1.0 / (1.0 + d),
    "l2": lambda d: 1.0 / (1.0 + d),
}


class InfinoVectorStore(VectorStore):
    """LangChain ``VectorStore`` backed by a single Infino table.

    The table holds the document id, the text, and the embedding, plus any
    promoted metadata columns and a JSON catch-all for the rest. The id and
    text columns are FTS-indexed: the id so ``get_by_ids`` resolves via
    ``exact_match`` (the engine's only pre-I/O prune for random ids), the text
    so BM25 and hybrid retrieval run over the same data.

    Args:
        connection: a live :class:`infino.Connection`.
        table_name: the table to open (must already exist; use
            :meth:`from_texts` to create and populate one).
        embedding: the LangChain embeddings to use for query and documents.
        dim: embedding dimension; must match the table's vector column and
            lie in the engine's supported range [16, 4096].
        metric: distance metric to index with — ``"cosine"`` (default),
            ``"l2sq"`` / ``"l2"``, ``"negdot"`` / ``"dot"``. Relevance
            normalization is defined for cosine/l2/l2sq; others serve raw
            distances only.
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
        self._text_column = text_column
        self._vector_column = vector_column
        self._id_column = id_column
        self._metadata_columns = list(metadata_columns)
        self._metadata_column_names = [f.name for f in self._metadata_columns]
        # open_table only succeeds once a table holds data (its manifest lands
        # on first commit); from_texts passes the create_table handle directly.
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
        """Embed and add ``texts``, returning their ids.

        Caller-supplied ids are upserted (re-adding overwrites); omitted or
        gap ids are generated.
        """
        texts = list(texts)
        if not texts:
            return []

        # Superfiles are immutable, so upsert = delete-then-append. Generated
        # uuids can't collide, so the delete is skipped on the bulk-load path.
        ids_provided = ids is not None
        if ids is None:
            ids = [uuid4().hex for _ in texts]
        elif len(ids) != len(texts):
            raise ValueError("ids and texts must have the same length")
        else:
            ids = [i if i is not None else uuid4().hex for i in ids]
        if ids_provided:
            self.delete(ids)

        if metadatas is None:
            metadatas = [{} for _ in texts]
        elif len(metadatas) != len(texts):
            raise ValueError("metadatas and texts must have the same length")

        vectors = self._embedding.embed_documents(texts)
        declared = set(self._metadata_column_names)

        # Order must match the schema: id, text, vector, *metadata, json.
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
        *,
        filter_query: str | None = None,
        filter_column: str | None = None,
        filter_mode: str | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        embedding = self._embedding.embed_query(query)
        return self.similarity_search_by_vector(
            embedding,
            k,
            filter=filter,
            filter_query=filter_query,
            filter_column=filter_column,
            filter_mode=filter_mode,
            **kwargs,
        )

    def similarity_search_by_vector(
        self,
        embedding: Sequence[float],
        k: int = DEFAULT_K,
        filter: Mapping[str, Any] | None = None,
        *,
        filter_query: str | None = None,
        filter_column: str | None = None,
        filter_mode: str | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        results = self._search(
            list(embedding),
            k,
            filter,
            filter_query=filter_query,
            filter_column=filter_column,
            filter_mode=filter_mode,
        )
        return [doc for doc, _ in results]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = DEFAULT_K,
        filter: Mapping[str, Any] | None = None,
        *,
        filter_query: str | None = None,
        filter_column: str | None = None,
        filter_mode: str | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        embedding = self._embedding.embed_query(query)
        results = self._search(
            embedding,
            k,
            filter,
            filter_query=filter_query,
            filter_column=filter_column,
            filter_mode=filter_mode,
        )
        return [(doc, score if score is not None else 0.0) for doc, score in results]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = DEFAULT_K,
        fetch_k: int = DEFAULT_FETCH_K,
        lambda_mult: float = DEFAULT_LAMBDA_MULT,
        filter: Mapping[str, Any] | None = None,
        *,
        filter_query: str | None = None,
        filter_column: str | None = None,
        filter_mode: str | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        # Stored vectors can't be read back (not projectable, no point-lookup),
        # so re-embed the candidate text for MMR's pairwise scoring.
        query_embedding = self._embedding.embed_query(query)
        candidates = self._search(
            query_embedding,
            fetch_k,
            filter,
            filter_query=filter_query,
            filter_column=filter_column,
            filter_mode=filter_mode,
        )
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

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Fetch documents by ``doc_id`` via ``exact_match`` (the only pre-I/O
        prune for random ids on a scan-based engine).

        Missing ids are skipped and duplicates collapse; order is not
        guaranteed, per the ``VectorStore`` contract.
        """
        projection = [
            self._id_column,
            self._text_column,
            *self._metadata_column_names,
            METADATA_JSON_COLUMN,
        ]
        found: dict[str, Document] = {}
        for id_ in ids:
            result = self._table.exact_match(
                self._id_column, id_, projection=projection
            )
            for doc, _ in rows_to_documents(
                result, id_column=self._id_column, text_column=self._text_column
            ):
                if doc.id is not None:
                    found[doc.id] = doc
        return list(found.values())

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
        *,
        filter_query: str | None = None,
        filter_column: str | None = None,
        filter_mode: str | None = None,
    ) -> list[tuple[Document, float | None]]:
        # Not composable in one engine call: `filter` is a post-rank SQL WHERE,
        # `filter_query` an FTS pre-filter the kNN honors before ranking.
        if filter and filter_query:
            raise ValueError(
                "pass either `filter` (structured SQL predicate, post-rank) or "
                "`filter_query` (text pushdown pre-filter), not both"
            )
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
        elif filter_query is not None:
            # Pushdown: engine prunes to FTS matches before ranking, so exactly
            # k are scored among survivors. Defaults to the indexed text column.
            result = self._table.vector_search(
                self._vector_column,
                list(embedding),
                k,
                filter_column=filter_column or self._text_column,
                filter_query=filter_query,
                filter_mode=filter_mode,
                projection=projection,
            )
        else:
            result = self._table.vector_search(
                self._vector_column, list(embedding), k, projection=projection
            )
        return rows_to_documents(
            result, id_column=self._id_column, text_column=self._text_column
        )

    def _to_documents(self, result: pa.Table) -> list[Document]:
        return [
            doc
            for doc, _ in rows_to_documents(
                result, id_column=self._id_column, text_column=self._text_column
            )
        ]

    def _hybrid_search(self, query: str, k: int = DEFAULT_K) -> list[Document]:
        """BM25 + vector retrieval fused by RRF in a single SQL call."""
        query_vector = _vector_literal(self._embedding.embed_query(query))
        # Project explicitly (SELECT * would leak engine-internal columns into
        # metadata). RRF score is larger-is-better, hence DESC.
        columns = ", ".join(self._projection())
        sql = (
            f"SELECT {columns} FROM hybrid_search("
            f"{sql_lit(self._table_name)}, {sql_lit(self._text_column)}, "
            f"{sql_lit(query)}, {sql_lit(self._vector_column)}, "
            f"{sql_lit(query_vector)}, {k}) ORDER BY {SCORE_COLUMN} DESC"
        )
        return self._to_documents(self._connection.query_sql(sql))

    def _bm25_search(
        self, query: str, k: int = DEFAULT_K, mode: str | None = None
    ) -> list[Document]:
        """Lexical BM25 retrieval over the FTS-indexed text column."""
        result = self._table.bm25_search(
            self._text_column, query, k, mode=mode, projection=self._projection()
        )
        return self._to_documents(result)

    def search_by_sql(self, sql: str) -> list[Document]:
        """Run arbitrary SQL over the engine and map the rows to documents.

        The escape hatch for what the typed methods don't cover — joins,
        custom ``WHERE``, or the ``vector_search`` / ``hybrid_search`` TVFs.
        Project the store's columns (id, text, declared metadata,
        ``_metadata_json``, and optionally ``score``) for full documents.
        """
        return self._to_documents(self._connection.query_sql(sql))

    def as_hybrid_retriever(self, k: int = DEFAULT_K) -> InfinoHybridRetriever:
        """A retriever that fuses BM25 and vector search (RRF) per query."""
        from langchain_infino.retrievers import InfinoHybridRetriever

        return InfinoHybridRetriever(vectorstore=self, k=k)

    def as_bm25_retriever(
        self, k: int = DEFAULT_K, mode: str | None = None
    ) -> InfinoBM25Retriever:
        """A lexical BM25 retriever over the text column."""
        from langchain_infino.retrievers import InfinoBM25Retriever

        return InfinoBM25Retriever(vectorstore=self, k=k, mode=mode)

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
    """Compile a structured metadata filter to a SQL ``WHERE`` clause.

    Supports equality, the operator form (``$gt`` / ``$in`` / ...), and
    ``$and`` / ``$or`` / ``$not``. A non-declared column is rejected — the
    engine can't index into the serialized JSON catch-all.
    """
    return _compile_node(filter, set(allowed))


def _compile_node(node: Mapping[str, Any], allowed: set[str]) -> str:
    clauses: list[str] = []
    for key, condition in node.items():
        if key in _LOGICAL_OPERATORS:
            joiner = _LOGICAL_OPERATORS[key]
            sub = [f"({_compile_node(f, allowed)})" for f in condition]
            clauses.append("(" + joiner.join(sub) + ")")
        elif key == "$not":
            inner = condition[0] if isinstance(condition, (list, tuple)) else condition
            clauses.append(f"NOT ({_compile_node(inner, allowed)})")
        else:
            if key not in allowed:
                raise ValueError(
                    f"cannot filter on {key!r}: not a declared metadata column "
                    f"(declared: {sorted(allowed)})"
                )
            clauses.append(_compile_comparison(key, condition))
    return " AND ".join(clauses)


def _compile_comparison(key: str, condition: Any) -> str:
    if not isinstance(condition, Mapping):
        return f"{key} = {_filter_literal(condition)}"
    parts: list[str] = []
    for op, value in condition.items():
        if op in ("$in", "$nin"):
            items = ", ".join(_filter_literal(v) for v in value)
            sql_op = "IN" if op == "$in" else "NOT IN"
            parts.append(f"{key} {sql_op} ({items})")
        elif op in _FILTER_OPERATORS:
            parts.append(f"{key} {_FILTER_OPERATORS[op]} {_filter_literal(value)}")
        else:
            raise ValueError(f"unsupported filter operator {op!r}")
    return " AND ".join(parts)


def _filter_literal(value: Any) -> str:
    """Render a filter value as a SQL literal."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return sql_lit(value)
    raise ValueError(f"unsupported filter value type: {type(value).__name__}")
