"""The :class:`InfinoVectorStore` LangChain vector store.

Maps the ``VectorStore`` contract onto one Infino table: text and embedding
in dedicated columns, the doc id in an FTS-indexed column, metadata either
promoted to scalar columns (filterable) or kept in a JSON catch-all. Vector,
filtered, MMR, and hybrid (RRF) retrieval all run over that one table.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Callable, Literal
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

# Mirror the engine's accepted values so the types flow through unchanged.
Metric = Literal["cosine", "l2sq", "l2", "negdot", "dot"]
SearchMode = Literal["or", "and"]
Bm25Stats = Literal["per_superfile", "global"]
ColdFetchMode = Literal[
    "hybrid_with_prefetch",
    "range_only",
    "lazy_foreground_with_background_fill",
]

DEFAULT_K = 4
DEFAULT_METRIC: Metric = "cosine"
DEFAULT_TEXT_COLUMN = "page_content"
DEFAULT_VECTOR_COLUMN = "embedding"
DEFAULT_ID_COLUMN = "doc_id"
DEFAULT_TABLE_NAME = "langchain"
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
    # Squared-L2 is unbounded above; map monotonically into (0, 1]. An exact
    # match can score a hair below zero in float, so floor the distance.
    "l2sq": lambda d: 1.0 / (1.0 + max(0.0, d)),
    "l2": lambda d: 1.0 / (1.0 + max(0.0, d)),
}



# Vector tuning knobs removed with engine-decided serving (infino#546).
# LangChain's `**kwargs` convention would swallow them silently; fail loud
# with a migration hint instead.
_REMOVED_KNOBS = ("nprobe", "rerank_mult")


def _reject_removed_knobs(kwargs: Mapping[str, Any]) -> None:
    for name in _REMOVED_KNOBS:
        if name in kwargs:
            raise TypeError(
                f"`{name}` was removed: vector serving (probe width and "
                "rerank budget) is engine-decided, calibrated per table at "
                "optimize time; drop the argument"
            )



def _l2_normalize(vectors: list[list[float]]) -> list[list[float]]:
    """Unit-normalize for the cosine metric.

    The engine's cosine contract expects unit-ish inputs: the stored rerank
    payload lives on a fixed [-1, 1] grid, so an unnormalized component
    clamps and distorts served distances (an exact self-match measured
    0.07 instead of ~0.0 with raw Gaussian embeddings). Cosine is
    scale-invariant, so normalizing changes nothing semantically — it only
    keeps every component on the representable grid.
    """
    out: list[list[float]] = []
    for v in vectors:
        norm = math.sqrt(sum(x * x for x in v))
        out.append([x / norm for x in v] if norm > 0.0 else list(v))
    return out


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
        dim: embedding dimension. Read off the table's vector column when
            omitted; if given, it must match. The engine supports [16, 4096].
        metric: distance metric to index with — ``"cosine"`` (default),
            ``"l2sq"`` / ``"l2"``, ``"negdot"`` / ``"dot"``. Relevance
            normalization is defined for cosine/l2/l2sq; others serve raw
            distances only.
        text_column / vector_column / id_column: column names.
        metadata_columns: metadata keys promoted to real scalar columns, so
            they can be filtered with the ``filter`` argument; the rest is
            kept in the JSON catch-all, which is not filterable. Taken from
            the table's own schema when omitted. Fixed at table creation —
            adding a filterable key later means recreating the table.
    """

    def __init__(
        self,
        connection: infino.Connection,
        table_name: str,
        embedding: Embeddings,
        *,
        dim: int | None = None,
        metric: Metric = DEFAULT_METRIC,
        text_column: str = DEFAULT_TEXT_COLUMN,
        vector_column: str = DEFAULT_VECTOR_COLUMN,
        id_column: str = DEFAULT_ID_COLUMN,
        metadata_columns: Sequence[pa.Field] | None = None,
        table: infino.Table | None = None,
    ) -> None:
        self._connection = connection
        self._table_name = table_name
        self._embedding = embedding
        self._metric = metric
        self._text_column = text_column
        self._vector_column = vector_column
        self._id_column = id_column
        # `table` is the handle from a just-created table, passed straight
        # through to save a redundant open; without one, open the named table.
        self._table = table if table is not None else connection.open_table(table_name)

        # The table already declares its embedding width and which metadata
        # keys were promoted, so neither has to be repeated to open one. Read
        # the schema only if one of them is actually missing.
        if dim is None or metadata_columns is None:
            schema = self._table.schema()
            if dim is None:
                dim = _dim_from_schema(schema, vector_column)
            if metadata_columns is None:
                metadata_columns = _metadata_fields_from_schema(
                    schema,
                    text_column=text_column,
                    vector_column=vector_column,
                    id_column=id_column,
                )
        self._dim = dim
        self._metadata_columns = list(metadata_columns)
        self._metadata_column_names = [f.name for f in self._metadata_columns]

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    @property
    def connection(self) -> infino.Connection:
        """The underlying connection — for engine calls this class doesn't wrap
        (``list_tables``, ``create_database``, cross-table SQL)."""
        return self._connection

    @property
    def table(self) -> infino.Table:
        """The underlying table — for engine calls this class doesn't wrap."""
        return self._table

    @property
    def table_name(self) -> str:
        return self._table_name

    @property
    def metric(self) -> Metric:
        return self._metric

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def metadata_columns(self) -> list[pa.Field]:
        """The promoted metadata columns — the keys ``filter`` can reach."""
        return list(self._metadata_columns)

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
        if self._metric == "cosine":
            vectors = _l2_normalize(vectors)
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
        filter_mode: SearchMode | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        _reject_removed_knobs(kwargs)
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
        filter_mode: SearchMode | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        _reject_removed_knobs(kwargs)
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
        filter_mode: SearchMode | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        _reject_removed_knobs(kwargs)
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
        filter_mode: SearchMode | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        _reject_removed_knobs(kwargs)
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
        projection = self._projection(score=False)
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

    def _projection(self, *, score: bool = True) -> list[str]:
        """Document columns to project.

        Only the ranking searches append ``score``; projecting it from
        ``exact_match`` / ``token_match`` is an unknown-column error.
        """
        columns = [
            self._id_column,
            self._text_column,
            *self._metadata_column_names,
            METADATA_JSON_COLUMN,
        ]
        if score:
            columns.append(SCORE_COLUMN)
        return columns

    def _search(
        self,
        embedding: Sequence[float],
        k: int,
        filter: Mapping[str, Any] | None = None,
        *,
        filter_query: str | None = None,
        filter_column: str | None = None,
        filter_mode: SearchMode | None = None,
    ) -> list[tuple[Document, float | None]]:
        if self._metric == "cosine":
            embedding = _l2_normalize([list(embedding)])[0]
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
                self._vector_column,
                list(embedding),
                k,
                projection=projection,
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

    def _hybrid_search(
        self,
        query: str,
        k: int = DEFAULT_K,
    ) -> list[Document]:
        """BM25 + vector retrieval fused by RRF in one engine call."""
        query_vector = self._embedding.embed_query(query)
        if self._metric == "cosine":
            query_vector = _l2_normalize([query_vector])[0]
        result = self._table.hybrid_search(
            self._text_column,
            query,
            self._vector_column,
            query_vector,
            k,
            projection=self._projection(),
        )
        return self._to_documents(result)

    def _bm25_search(
        self,
        query: str,
        k: int = DEFAULT_K,
        mode: SearchMode | None = None,
        *,
        stats: Bm25Stats | None = None,
    ) -> list[Document]:
        """Lexical BM25 retrieval over the FTS-indexed text column."""
        result = self._table.bm25_search(
            self._text_column,
            query,
            k,
            mode=mode,
            stats=stats,
            projection=self._projection(),
        )
        return self._to_documents(result)

    def token_search(
        self,
        query: str,
        *,
        column: str | None = None,
        mode: SearchMode | None = None,
    ) -> list[Document]:
        """Every document whose ``column`` matches ``query``'s terms.

        Set membership, not relevance — hence no ``k`` and no score. Matches
        against the text column unless told otherwise. For ranked top-k, use
        :meth:`as_bm25_retriever`.
        """
        result = self._table.token_match(
            column or self._text_column,
            query,
            mode=mode,
            projection=self._projection(score=False),
        )
        return self._to_documents(result)

    def exact_search(self, value: str, column: str) -> list[Document]:
        """Documents whose ``column`` equals ``value`` verbatim.

        No tokenization, no ranking. ``column`` must be FTS-indexed — the
        store indexes the id and text columns, nothing else.
        """
        result = self._table.exact_match(
            column, value, projection=self._projection(score=False)
        )
        return self._to_documents(result)

    def count(
        self,
        query: str,
        *,
        column: str | None = None,
        mode: SearchMode | None = None,
    ) -> int:
        """How many documents match ``query``, without materializing rows.

        Counted in the engine, so it stays cheap on a table larger than
        memory. This counts term matches; for a whole-table count run
        ``SELECT COUNT(*)`` through :attr:`connection`.
        """
        return self._table.count(column or self._text_column, query, mode=mode)

    def optimize(
        self,
        *,
        max_memory_mb: int | None = None,
        min_fill_percent: int | None = None,
        target_superfile_size_mb: int | None = None,
        stale_seal_timeout_ms: int | None = None,
    ) -> None:
        """Compact the table and recalibrate its vector serving.

        Appends land as small immutable superfiles; this merges them, and is
        also where the engine calibrates vector serving for the table's
        current size. Search works without it, but recall and latency both
        improve once it has run.

        The arguments bound the work; omit them for the engine's defaults.
        """
        settings = None
        if any(
            v is not None
            for v in (
                max_memory_mb,
                min_fill_percent,
                target_superfile_size_mb,
                stale_seal_timeout_ms,
            )
        ):
            settings = infino.OptimizeOptions(
                max_memory_mb=max_memory_mb,
                min_fill_percent=min_fill_percent,
                target_superfile_size_mb=target_superfile_size_mb,
                stale_seal_timeout_ms=stale_seal_timeout_ms,
            )
        self._table.optimize(settings)

    def gc(self, grace_secs: float) -> infino.GcReport:
        """Delete storage objects no live snapshot references.

        ``grace_secs`` spares anything younger, so readers still on an older
        snapshot are not pulled out from under.
        """
        return self._table.gc(grace_secs)

    def schema(self) -> pa.Schema:
        """The table's declared Arrow schema."""
        return self._table.schema()

    def search_by_sql(self, sql: str) -> list[Document]:
        """Run arbitrary SQL over the engine and map the rows to documents.

        The escape hatch for what the typed methods don't cover — joins,
        custom ``WHERE``, or the ``vector_search`` / ``hybrid_search`` TVFs.
        Project the store's columns (id, text, declared metadata,
        ``_metadata_json``, and optionally ``score``) for full documents.
        """
        return self._to_documents(self._connection.query_sql(sql))

    def as_hybrid_retriever(
        self,
        k: int = DEFAULT_K,
    ) -> InfinoHybridRetriever:
        """A retriever that fuses BM25 and vector search (RRF) per query."""
        from langchain_infino.retrievers import InfinoHybridRetriever

        return InfinoHybridRetriever(
            vectorstore=self, k=k
        )

    def as_bm25_retriever(
        self,
        k: int = DEFAULT_K,
        mode: SearchMode | None = None,
        *,
        stats: Bm25Stats | None = None,
    ) -> InfinoBM25Retriever:
        """A lexical BM25 retriever over the text column."""
        from langchain_infino.retrievers import InfinoBM25Retriever

        return InfinoBM25Retriever(vectorstore=self, k=k, mode=mode, stats=stats)

    @classmethod
    def open_or_create(
        cls,
        connection: infino.Connection,
        table_name: str,
        embedding: Embeddings,
        *,
        dim: int | None = None,
        metric: Metric = DEFAULT_METRIC,
        analyzer: str | None = None,
        text_column: str = DEFAULT_TEXT_COLUMN,
        vector_column: str = DEFAULT_VECTOR_COLUMN,
        id_column: str = DEFAULT_ID_COLUMN,
        metadata_columns: Sequence[pa.Field] | None = None,
    ) -> InfinoVectorStore:
        """Open ``table_name``, creating it first if it isn't there.

        Idempotent, unlike the plain constructor (which requires the table)
        and :meth:`from_texts` (which requires its absence).

        The creation arguments — ``metric``, ``analyzer``,
        ``metadata_columns`` — apply only when this call does the creating; an
        existing table keeps the schema it was made with, and a mismatch
        surfaces on the first write.
        """
        table: infino.Table | None = None
        if table_name not in connection.list_tables():
            if dim is None:
                dim = _dim_from_embedding(embedding)
            try:
                table = _create_table(
                    connection,
                    table_name,
                    dim=dim,
                    metric=metric,
                    analyzer=analyzer,
                    text_column=text_column,
                    vector_column=vector_column,
                    id_column=id_column,
                    metadata_columns=metadata_columns or (),
                )
            except ValueError:
                # Lost the race to a concurrent creator: use their table. Any
                # other rejection (an out-of-range `dim`, say) still raises.
                if table_name not in connection.list_tables():
                    raise
        return cls(
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

    @classmethod
    def connect(
        cls,
        uri: str,
        embedding: Embeddings,
        table_name: str,
        *,
        dim: int | None = None,
        create: bool = False,
        create_database: bool = False,
        # Connection configuration, mirroring the engine's own `connect`.
        storage_options: Mapping[str, str] | None = None,
        cache_dir: str | None = None,
        cache_budget_bytes: int | None = None,
        connection_memory_budget_bytes: int | None = None,
        cold_fetch_mode: ColdFetchMode | None = None,
        validate: bool | None = None,
        api_key: str | None = None,
        # Store configuration, as on `from_texts`.
        metric: Metric = DEFAULT_METRIC,
        analyzer: str | None = None,
        text_column: str = DEFAULT_TEXT_COLUMN,
        vector_column: str = DEFAULT_VECTOR_COLUMN,
        id_column: str = DEFAULT_ID_COLUMN,
        metadata_columns: Sequence[pa.Field] | None = None,
    ) -> InfinoVectorStore:
        """Open a store straight from a URI, without building a connection first.

        One call from a storage location to a usable store — a local
        directory, cloud object storage, or a hosted Infino target:

        .. code-block:: python

            # local directory
            InfinoVectorStore.connect("./data", emb, "docs", dim=768)

            # object storage
            InfinoVectorStore.connect(
                "s3://bucket/prefix", emb, "docs", dim=768,
                storage_options={"aws_region": "us-east-1"},
            )

            # hosted
            InfinoVectorStore.connect(
                "https://...", emb, "docs", dim=768, api_key="...",
            )

        ``create`` ensures the table exists — created when absent, opened
        when present — so it is safe on every run; left false, the table must
        already exist. ``create_database`` provisions the database first,
        which a hosted target needs once.

        The connection options are the engine's own, forwarded unchanged;
        only those explicitly set are sent, so the rest keep engine defaults.
        ``storage_options`` carries backend credentials and settings, and
        ambient credentials need none. The remaining arguments configure the
        table as on :meth:`open_or_create`.
        """
        options: dict[str, Any] = {}
        if storage_options is not None:
            options["storage_options"] = dict(storage_options)
        if cache_dir is not None:
            options["cache_dir"] = cache_dir
        if cache_budget_bytes is not None:
            options["cache_budget_bytes"] = cache_budget_bytes
        if connection_memory_budget_bytes is not None:
            options["connection_memory_budget_bytes"] = connection_memory_budget_bytes
        if cold_fetch_mode is not None:
            options["cold_fetch_mode"] = cold_fetch_mode
        if validate is not None:
            options["validate"] = validate
        if api_key is not None:
            options["api_key"] = api_key
        connection = infino.connect(uri, **options)

        if create_database:
            connection.create_database()

        if create:
            return cls.open_or_create(
                connection,
                table_name,
                embedding,
                dim=dim,
                metric=metric,
                analyzer=analyzer,
                text_column=text_column,
                vector_column=vector_column,
                id_column=id_column,
                metadata_columns=metadata_columns,
            )
        return cls(
            connection,
            table_name,
            embedding,
            dim=dim,
            metric=metric,
            text_column=text_column,
            vector_column=vector_column,
            id_column=id_column,
            metadata_columns=metadata_columns,
        )

    @classmethod
    def from_texts(  # type: ignore[override]  # requires engine params (connection, table_name, dim) the base signature lacks
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        *,
        connection: infino.Connection,
        table_name: str = DEFAULT_TABLE_NAME,
        dim: int | None = None,
        ids: list[str] | None = None,
        metric: Metric = DEFAULT_METRIC,
        analyzer: str | None = None,
        text_column: str = DEFAULT_TEXT_COLUMN,
        vector_column: str = DEFAULT_VECTOR_COLUMN,
        id_column: str = DEFAULT_ID_COLUMN,
        metadata_columns: Sequence[pa.Field] | None = None,
        **kwargs: Any,
    ) -> InfinoVectorStore:
        """Create the table, then embed and insert ``texts``.

        ``dim`` is taken from ``embedding`` when omitted. ``metadata_columns``
        likewise defaults to the scalar keys found in ``metadatas``, promoting
        them to filterable columns; pass ``()`` to promote none, or an explicit
        list to control names, types and nullability. Keys that are nested or
        inconsistently typed stay in the JSON catch-all either way.
        """
        if dim is None:
            dim = _dim_from_embedding(embedding)
        if metadata_columns is None:
            metadata_columns = _infer_metadata_columns(
                metadatas or [],
                _reserved_columns(text_column, vector_column, id_column),
            )
        table = _create_table(
            connection,
            table_name,
            dim=dim,
            metric=metric,
            analyzer=analyzer,
            text_column=text_column,
            vector_column=vector_column,
            id_column=id_column,
            metadata_columns=metadata_columns,
        )

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


def _dim_from_embedding(embedding: Embeddings) -> int:
    """Measure the embedding width, for a table that has yet to declare one."""
    return len(embedding.embed_query("dimension probe"))


def _dim_from_schema(schema: pa.Schema, vector_column: str) -> int:
    """Read the embedding width off a table's declared vector column."""
    try:
        field = schema.field(vector_column)
    except KeyError:
        raise ValueError(
            f"table has no vector column {vector_column!r} "
            f"(columns: {schema.names}); pass `vector_column=`"
        ) from None
    size = getattr(field.type, "list_size", None)
    if size is None:
        raise ValueError(
            f"column {vector_column!r} is {field.type}, not a fixed-size list; "
            f"pass `dim=` explicitly"
        )
    return int(size)


def _metadata_fields_from_schema(
    schema: pa.Schema,
    *,
    text_column: str,
    vector_column: str,
    id_column: str,
) -> list[pa.Field]:
    """The promoted metadata columns of an existing table.

    Whatever is neither an id, text, vector nor the JSON catch-all was
    promoted at creation, so it is filterable.
    """
    reserved = {id_column, text_column, vector_column, METADATA_JSON_COLUMN}
    return [field for field in schema if field.name not in reserved]


def _reserved_columns(
    text_column: str, vector_column: str, id_column: str
) -> set[str]:
    """Column names a promoted metadata key may not take.

    ``score`` is the relevance column the search functions append and ``_id``
    the engine's internal key; colliding with either produces a duplicate
    column the engine rejects mid-query.
    """
    return {
        text_column,
        vector_column,
        id_column,
        METADATA_JSON_COLUMN,
        SCORE_COLUMN,
        "_id",
    }


def _infer_metadata_columns(
    metadatas: Sequence[Mapping[str, Any]],
    reserved: set[str],
) -> list[pa.Field]:
    """Promote the scalar metadata keys in ``metadatas`` to real columns.

    A key is promoted only if every value it carries is a scalar of one
    consistent type; anything mixed, nested, or named like a structural column
    stays in the JSON catch-all, which is not filterable. Promoted columns are
    nullable because the schema is fixed at creation and a later append may
    omit the key.
    """
    types: dict[str, type] = {}
    skipped: set[str] = set(reserved)
    for metadata in metadatas:
        for key, value in metadata.items():
            if value is None or key in skipped:
                continue
            # bool before int: bool is an int subclass in Python.
            for python_type in (bool, int, float, str):
                if isinstance(value, python_type):
                    break
            else:
                skipped.add(key)
                types.pop(key, None)
                continue
            known = types.setdefault(key, python_type)
            if known is not python_type:
                # int alongside float widens; anything else is a real conflict.
                if {known, python_type} == {int, float}:
                    types[key] = float
                else:
                    skipped.add(key)
                    types.pop(key, None)

    arrow_types = {
        bool: pa.bool_(),
        int: pa.int64(),
        float: pa.float64(),
        str: pa.large_utf8(),
    }
    return [
        pa.field(key, arrow_types[python_type], nullable=True)
        for key, python_type in types.items()
    ]


def _create_table(
    connection: infino.Connection,
    table_name: str,
    *,
    dim: int,
    metric: Metric,
    analyzer: str | None,
    text_column: str,
    vector_column: str,
    id_column: str,
    metadata_columns: Sequence[pa.Field],
) -> infino.Table:
    """Create the table with its schema and indexes.

    The id column keeps the default analyzer so ``exact_match`` resolves ids
    verbatim, whatever ``analyzer`` the text column gets.
    """
    reserved = _reserved_columns(text_column, vector_column, id_column) - {
        text_column,
        vector_column,
        id_column,
    }
    clashing = sorted(f.name for f in metadata_columns if f.name in reserved)
    if clashing:
        raise ValueError(
            f"metadata column(s) {clashing} collide with columns the engine "
            f"reserves ({sorted(reserved)}); rename them"
        )
    return connection.create_table(
        table_name,
        _build_schema(dim, text_column, vector_column, id_column, metadata_columns),
        infino.IndexSpec()
        .fts(text_column, analyzer)
        .fts(id_column)
        .vector(vector_column, dim, metric),
    )


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
