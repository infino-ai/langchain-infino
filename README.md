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

Early. Phase 1 implements the core `VectorStore` contract:

- `from_texts`, `add_texts`, `delete`
- `similarity_search`, `similarity_search_with_score`, `similarity_search_by_vector`
- `as_retriever` (inherited)

Metadata is round-tripped through a JSON catch-all column. Planned next:
declared filterable metadata columns, a hybrid (RRF) retriever in a single
SQL call, MMR, and a self-query translator.

## Development

```bash
make install      # pip install -e ".[test,lint]"
make unit         # unit tests (no engine)
make integration  # integration tests (real Infino on a temp dir)
make lint type    # ruff + mypy
```

## License

Apache-2.0.
