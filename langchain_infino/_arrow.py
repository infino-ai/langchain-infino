"""Helpers bridging Infino's Arrow results and LangChain documents.

Kept in one module because the projection column names and the SQL
literal-quoting rules are schema assumptions that must stay in a single
place — they are the contract between :class:`InfinoVectorStore` and the
engine.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pyarrow as pa
from langchain_core.documents import Document

# Catch-all column holding the JSON-serialized remainder of a document's
# metadata (everything not promoted to a declared scalar column).
METADATA_JSON_COLUMN = "_metadata_json"

# Trailing relevance column every search TVF appends.
SCORE_COLUMN = "score"


def sql_lit(value: str) -> str:
    """Quote a string as a SQL literal, escaping embedded single quotes.

    Infino predicates are built as SQL text (e.g. ``doc_id IN (...)``), so
    any caller-supplied id or term must be escaped to avoid breaking the
    statement. Numeric literals (vectors) never pass through here.
    """
    return "'" + value.replace("'", "''") + "'"


def serialize_metadata(metadata: Mapping[str, Any]) -> str:
    """Serialize a document's metadata dict into the JSON catch-all column."""
    return json.dumps(metadata, separators=(",", ":"), sort_keys=True)


def vector_array(vectors: list[list[float]], dim: int) -> pa.Array:
    """Pack embedding vectors into a ``fixed_size_list<float32, dim>`` array."""
    return pa.array(vectors, type=pa.list_(pa.float32(), dim))


def rows_to_documents(
    table: pa.Table,
    *,
    id_column: str,
    text_column: str,
) -> list[tuple[Document, float | None]]:
    """Convert a search result table into ``(Document, score)`` pairs.

    ``score`` is ``None`` when the projection did not include it. The id is
    folded back into ``Document.metadata`` under ``id_column`` so callers can
    recover it; the JSON catch-all is merged in underneath.
    """
    n = table.num_rows
    if n == 0:
        return []

    columns = {name: table.column(name).to_pylist() for name in table.column_names}
    ids = columns.get(id_column, [None] * n)
    texts = columns.get(text_column, [""] * n)
    metadata_json = columns.get(METADATA_JSON_COLUMN, [None] * n)
    scores = columns.get(SCORE_COLUMN, [None] * n)

    results: list[tuple[Document, float | None]] = []
    for i in range(n):
        raw = metadata_json[i]
        metadata: dict[str, Any] = json.loads(raw) if raw else {}
        metadata[id_column] = ids[i]
        doc = Document(page_content=texts[i] or "", metadata=metadata)
        results.append((doc, scores[i]))
    return results
