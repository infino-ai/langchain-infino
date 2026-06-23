# langchain-infino

[![PyPI](https://img.shields.io/pypi/v/langchain-infino.svg)](https://pypi.org/project/langchain-infino/)
[![Python](https://img.shields.io/pypi/pyversions/langchain-infino.svg)](https://pypi.org/project/langchain-infino/)
[![License](https://img.shields.io/pypi/l/langchain-infino.svg)](https://www.apache.org/licenses/LICENSE-2.0)

**LangChain over [Infino](https://github.com/infino-ai/infino) — vector,
full-text (BM25), hybrid, and SQL-native retrieval over one copy of your data
on object storage.**

Most "vector database" LangChain integrations expose only the vector slice of
their engine. Infino keeps your data in Apache Parquet on object storage and
runs SQL, BM25, vector, and hybrid (RRF) retrieval over it from a single
in-process engine — no separate search cluster or vector store to keep in
sync. This package surfaces that whole retrieval surface, not just
`similarity_search`.

Infino never embeds: you bring a LangChain `Embeddings` object, and the
integration supplies the vectors.

## Installation

```sh
pip install langchain-infino
```

Or with [uv](https://docs.astral.sh/uv/):

```sh
uv add langchain-infino
```

Requires Python 3.9+. `infino`, `langchain-core`, `pyarrow`, and `numpy` are
installed as dependencies. Bring your own embeddings provider separately (e.g.
`pip install langchain-openai`).

## Quickstart

```python
import infino
from langchain_infino import InfinoVectorStore
from langchain_openai import OpenAIEmbeddings

# A local path or an S3 URI for durable storage; "memory://" is ephemeral.
connection = infino.connect("./data")
embedding = OpenAIEmbeddings()  # dim must match the table; 1536 here

store = InfinoVectorStore.from_texts(
    ["Infino runs search on object storage.", "One engine for SQL, BM25, and vectors."],
    embedding,
    connection=connection,
    table_name="docs",
    dim=1536,
)

docs = store.similarity_search("search on S3", k=2)
retriever = store.as_retriever()
```

## Core concepts

- **`InfinoVectorStore`** wraps a single Infino table — the text, its
  embedding, the document id, declared metadata columns, and a JSON catch-all.
  Use `from_texts` to create and populate one; construct directly to open an
  existing table.
- **Identity** — caller-controlled ids live on `Document.id` (not in
  metadata). `add_texts` is an idempotent upsert: re-adding an id overwrites,
  omitted ids are generated.
- **Metadata, two tiers** — keys you name in `metadata_columns=` become real
  scalar columns you can filter on; everything else round-trips losslessly
  through a JSON catch-all but isn't filterable. The schema is fixed at table
  creation — adding a filterable key means recreating the table.
- **Scores** — vector distance is *smaller is nearer*; BM25 and RRF are
  *larger is better*. `similarity_search_with_relevance_scores` normalizes to
  `[0, 1]` (higher = better) for `cosine`, `l2`, and `l2sq`.
- **Retrievers** — `as_retriever()` (vector), `as_bm25_retriever()` (lexical),
  and `as_hybrid_retriever()` (RRF fusion).
- **Dimensions** — embeddings must be `[16, 4096]`-dimensional (engine limit)
  and match the table's declared `dim`.

## Adding and managing documents

```python
# Generated ids on the common path; returns them.
ids = store.add_texts(["a new note"], metadatas=[{"source": "inbox"}])

# Caller ids are upserted — re-adding "doc-1" overwrites in place.
store.add_texts(["v2 of the note"], ids=["doc-1"])

# Fetch by id (skips missing, order not guaranteed); delete by id.
store.get_by_ids(["doc-1"])
store.delete(["doc-1"])
```

## Similarity search

```python
store.similarity_search("vector databases", k=4)
store.similarity_search_with_score("vector databases", k=4)       # raw distance
store.similarity_search_with_relevance_scores("vector databases", k=4)  # [0, 1]
store.similarity_search_by_vector(query_vector, k=4)              # query_vector: list[float]
```

## Metadata filtering

Promote the keys you want to filter on to real columns, then pass the
LangChain operator form. Supports equality, `$eq` / `$ne` / `$gt` / `$gte` /
`$lt` / `$lte`, `$in` / `$nin`, and `$and` / `$or` / `$not`.

```python
import pyarrow as pa

store = InfinoVectorStore.from_texts(
    texts, embedding,
    connection=connection, table_name="papers", dim=1536,
    metadata_columns=[
        pa.field("category", pa.large_utf8(), nullable=False),
        pa.field("year", pa.int64(), nullable=False),
    ],
    metadatas=[{"category": "ml", "year": 2024} for _ in texts],
)

store.similarity_search("optimizers", k=4, filter={"category": "ml"})
store.similarity_search("optimizers", k=4, filter={"year": {"$gte": 2023}})
store.similarity_search("optimizers", k=4,
                        filter={"$or": [{"category": "ml"}, {"year": {"$lt": 2000}}]})
```

## Text-pushdown pre-filter

For a *text* predicate, push it into the kNN instead of post-filtering the
top-k. The engine prunes to rows matching the full-text terms **before**
ranking, so exactly `k` nearest *matching* rows come back — no over-fetch, no
under-return. `filter_mode` is `"or"` (default) or `"and"`; `filter_column`
defaults to the text column.

```python
store.similarity_search("cancel my plan", k=10, filter_query="subscription billing")
```

It is reachable from any retriever via `search_kwargs`:

```python
retriever = store.as_retriever(search_kwargs={"k": 10, "filter_query": "billing"})
```

`filter` (structured, post-rank SQL `WHERE`) and `filter_query` (text,
pre-rank pushdown) are distinct paths and not combinable in one call.

## Maximal marginal relevance (MMR)

```python
store.max_marginal_relevance_search("transformers", k=4, fetch_k=20, lambda_mult=0.5)
```

Infino's vector column isn't projectable and there's no point-lookup, so MMR
re-embeds the `fetch_k` candidates' text to score them against each other.

## Hybrid (RRF) retrieval

BM25 and vector search fused by reciprocal-rank fusion in a single SQL call —
no separate reranking round-trip.

```python
retriever = store.as_hybrid_retriever(k=4)
retriever.invoke("neural network training")
```

## BM25 retrieval

Pure lexical ranking over the FTS-indexed text column.

```python
retriever = store.as_bm25_retriever(k=4)              # OR by default
retriever = store.as_bm25_retriever(k=4, mode="and")  # require all terms
retriever.invoke("gradient descent")
```

## Self-query

`InfinoTranslator` plugs into LangChain's `SelfQueryRetriever`, lowering an
LLM's structured query to a SQL `WHERE` over the declared metadata columns —
the full comparison and boolean surface, not a reduced DSL. Pass it as the
`structured_query_translator` (see LangChain's self-query docs for the
`metadata_field_info` setup):

```python
from langchain_infino import InfinoTranslator

retriever = SelfQueryRetriever.from_llm(
    llm,
    store,
    document_contents="research papers",
    metadata_field_info=metadata_field_info,
    structured_query_translator=InfinoTranslator(),
)
retriever.invoke("ML papers since 2023")
```

## SQL-native search

The escape hatch for anything the typed methods don't cover — joins, custom
`WHERE`, or the `vector_search` / `hybrid_search` table functions. Project the
store's columns (`doc_id`, `page_content`, declared metadata,
`_metadata_json`, and optionally `score`) and the rows map back to
`Document`s.

```python
qv = ",".join(map(str, embedding.embed_query("fox")))
store.search_by_sql(f"""
    SELECT doc_id, page_content, _metadata_json, score
    FROM hybrid_search('docs', 'page_content', 'fox', 'embedding', '{qv}', 10)
    ORDER BY score DESC
""")
```

## Semantic LLM cache

Caches model responses keyed by prompt *meaning*: a lookup embeds the prompt
and returns a hit when a stored prompt for the same model lands within a
distance threshold. One small Infino table, no extra infrastructure.

```python
from langchain_core.globals import set_llm_cache
from langchain_infino import InfinoSemanticCache

set_llm_cache(InfinoSemanticCache(connection, embedding, dim=1536))
```

## Async

The async methods (`aadd_texts`, `asimilarity_search`, …) are inherited from
`VectorStore`, which offloads the synchronous engine calls to a thread via
`run_in_executor` — the event loop is never blocked.

## API reference

- `InfinoVectorStore(connection, table_name, embedding, *, dim, metric="cosine", text_column="page_content", vector_column="embedding", id_column="doc_id", metadata_columns=())`
  — opens an existing table.
  - `from_texts(texts, embedding, metadatas=None, *, connection, table_name, dim, ids=None, metric="cosine", n_cent=64, text_column=..., vector_column=..., id_column=..., metadata_columns=()) -> InfinoVectorStore`
    — creates and populates the table.
  - `add_texts(texts, metadatas=None, *, ids=None) -> list[str]` — idempotent upsert.
  - `similarity_search(query, k=4, filter=None, *, filter_query=None, filter_column=None, filter_mode=None) -> list[Document]`
  - `similarity_search_with_score(...)`, `similarity_search_by_vector(...)`
  - `max_marginal_relevance_search(query, k=4, fetch_k=20, lambda_mult=0.5, filter=None, ...)`
  - `delete(ids) -> bool`, `get_by_ids(ids) -> list[Document]`
  - `search_by_sql(sql) -> list[Document]`
  - `as_retriever(...)`, `as_hybrid_retriever(k=4)`, `as_bm25_retriever(k=4, mode=None)`
- `InfinoHybridRetriever`, `InfinoBM25Retriever` — `BaseRetriever`s wrapping a store.
- `InfinoTranslator` — `StructuredQuery` → SQL filter, for `SelfQueryRetriever`.
- `InfinoSemanticCache(connection, embedding, *, dim, table_name="langchain_llm_cache", score_threshold=0.05)`

`metric` is `"cosine"` (default), `"l2sq"` / `"l2"`, or `"negdot"` / `"dot"`.
See [Infino](https://github.com/infino-ai/infino) for engine internals.

## Development

```sh
make install      # pip install -e ".[test,lint]"
make unit         # unit tests (no engine)
make integration  # integration + compliance tests (real Infino on a temp dir)
make lint type    # ruff + mypy
make build        # build sdist + wheel into dist/
make smoke        # build the wheel, install it in a clean venv, run the smoke test
make clean        # remove build artifacts and caches
```

## License

Apache-2.0.
</content>
