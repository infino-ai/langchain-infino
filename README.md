# langchain-infino

LangChain integration for [Infino](https://github.com/infino-ai/infino) — run
vector, full-text (BM25), and hybrid retrieval over **one copy of your data on
object storage**, from LangChain. No separate vector database to keep in sync.

```bash
pip install langchain-infino
```

## Quick start

```python
import infino
from langchain_infino import InfinoVectorStore
from langchain_openai import OpenAIEmbeddings

connection = infino.connect("./my_db")  # or an s3:// URI
embedding = OpenAIEmbeddings()

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

## Status

Early, but the core and the differentiators are in place.

**Core `VectorStore` contract** — passes LangChain's standard
`VectorStoreIntegrationTests` compliance suite.

- `from_texts`, `add_texts` (idempotent upsert on caller-supplied ids), `delete`
- `get_by_ids`
- `similarity_search`, `similarity_search_with_score`, `similarity_search_by_vector`
- `max_marginal_relevance_search`
- `as_retriever` (inherited)

**Infino-native extras**

- **Hybrid (RRF) retrieval** — `store.as_hybrid_retriever()` fuses BM25 and
  vector search in a single SQL call, no separate reranking step.
- **Metadata filtering** — promote metadata keys to real columns
  (`metadata_columns=`) and filter with the LangChain operator form
  (`filter={"year": {"$gte": 2023}}`, plus `$and` / `$or` / `$not`); the
  rest rides in a JSON catch-all.
- **Text-pushdown pre-filter** — `filter_query="..."` restricts the kNN to
  rows matching the full-text terms *before* ranking, so exactly `k` are
  scored among the survivors — no over-fetch, no under-return. `filter_mode`
  (`"or"` default / `"and"`) controls multi-term matching; `filter_column`
  defaults to the text column. Strictly better than post-filtering for text
  predicates, and reachable from a retriever via
  `as_retriever(search_kwargs={"filter_query": "..."})`.
- **Self-query** — `InfinoTranslator` plugs into `SelfQueryRetriever`, lowering
  an LLM's structured query to a SQL `WHERE` over the declared columns.
- **Semantic LLM cache** — `InfinoSemanticCache` matches prompts by meaning
  via vector search; a near-enough stored prompt returns the cached response.

Filtering on a hybrid query.

```python
import pyarrow as pa

store = InfinoVectorStore.from_texts(
    texts, embedding,
    connection=connection, table_name="docs", dim=1536,
    metadata_columns=[pa.field("category", pa.large_utf8(), nullable=False)],
    metadatas=[{"category": "ml"} for _ in texts],
)
docs = store.similarity_search("optimizers", k=4, filter={"category": "ml"})
```

## Development

```bash
make install      # pip install -e ".[test,lint]"
make unit         # unit tests (no engine)
make integration  # integration tests (real Infino on a temp dir)
make lint type    # ruff + mypy
```

## License

Apache-2.0.
