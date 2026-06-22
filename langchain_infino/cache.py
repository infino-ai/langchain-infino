"""A semantic LLM cache backed by Infino vector search.

Caches model responses keyed by the *meaning* of the prompt: a lookup
embeds the incoming prompt and returns a cached response when a stored
prompt for the same model lands within a distance threshold. One small
Infino table, no extra infrastructure.
"""

from __future__ import annotations

from typing import Any, cast

import infino
import pyarrow as pa
from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.embeddings import Embeddings
from langchain_core.load import dumps, loads

from langchain_infino.vectorstores import InfinoVectorStore

# Cosine distance below which a stored prompt counts as the same request.
DEFAULT_CACHE_DISTANCE_THRESHOLD = 0.05
# Nearest stored prompts to inspect for a matching model signature.
CACHE_LOOKUP_K = 5
# JSON catch-all key holding the serialized return value.
_RETURN_KEY = "_return"
_LLM_STRING_COLUMN = "llm_string"


class InfinoSemanticCache(BaseCache):
    """LangChain ``BaseCache`` doing semantic prompt matching over Infino.

    Args:
        connection: a live :class:`infino.Connection`.
        embedding: embeddings used to key prompts; must match ``dim``.
        dim: embedding dimension.
        table_name: cache table name.
        score_threshold: maximum cosine distance for a cache hit.
    """

    def __init__(
        self,
        connection: infino.Connection,
        embedding: Embeddings,
        *,
        dim: int,
        table_name: str = "langchain_llm_cache",
        score_threshold: float = DEFAULT_CACHE_DISTANCE_THRESHOLD,
    ) -> None:
        self._connection = connection
        self._embedding = embedding
        self._dim = dim
        self._table_name = table_name
        self._threshold = score_threshold
        self._store = self._build_store()

    def _build_store(self) -> InfinoVectorStore:
        return InfinoVectorStore.from_texts(
            [],
            self._embedding,
            connection=self._connection,
            table_name=self._table_name,
            dim=self._dim,
            metric="cosine",
            metadata_columns=[
                pa.field(_LLM_STRING_COLUMN, pa.large_utf8(), nullable=False)
            ],
        )

    def lookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        # k-NN by prompt meaning, then match the model signature in Python —
        # avoids a SQL filter path and works on an empty cache.
        for doc, distance in self._store.similarity_search_with_score(
            prompt, k=CACHE_LOOKUP_K
        ):
            if distance > self._threshold:
                break
            if doc.metadata.get(_LLM_STRING_COLUMN) == llm_string:
                # The cache only ever stores langchain_core Generations that
                # it serialized itself, so "core" is the safe allow-list.
                cached = loads(doc.metadata[_RETURN_KEY], allowed_objects="core")
                return cast(RETURN_VAL_TYPE, cached)
        return None

    def update(
        self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE
    ) -> None:
        self._store.add_texts(
            [prompt],
            metadatas=[
                {_LLM_STRING_COLUMN: llm_string, _RETURN_KEY: dumps(list(return_val))}
            ],
        )

    def clear(self, **kwargs: Any) -> None:
        self._connection.drop_table(self._table_name, purge=True)
        self._store = self._build_store()
