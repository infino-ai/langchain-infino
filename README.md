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

**Core `VectorStore` contract**

- `from_texts`, `add_texts`, `delete`
- `similarity_search`, `similarity_search_with_score`, `similarity_search_by_vector`
- `max_marginal_relevance_search`
- `as_retriever` (inherited)

**Infino-native extras**

- **Hybrid (RRF) retrieval** — `store.as_hybrid_retriever()` fuses BM25 and
  vector search in a single SQL call, no separate reranking step.
- **Metadata filtering** — promote metadata keys to real columns
  (`metadata_columns=`) and filter with the LangChain operator form
  (`filter={"year": {"$gte": 2023}}`); the rest rides in a JSON catch-all.

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

Planned next: a self-query translator and a semantic LLM cache.

## Development

```bash
make install      # pip install -e ".[test,lint]"
make unit         # unit tests (no engine)
make integration  # integration tests (real Infino on a temp dir)
make lint type    # ruff + mypy
```

## License

Apache-2.0.
