"""Helpers bridging Infino's Arrow results and LangChain documents.

One module so the projection column names and SQL-quoting rules — the schema
contract between the store and the engine — live in one place.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pyarrow as pa
from langchain_core.documents import Document

# Column holding metadata not promoted to a declared scalar column, as JSON.
METADATA_JSON_COLUMN = "_metadata_json"

# Trailing relevance column every search TVF appends.
SCORE_COLUMN = "score"


def sql_lit(value: str) -> str:
    """Quote a string as a SQL literal, escaping embedded single quotes.

    Infino predicates are built as SQL text, so caller-supplied ids/terms must
    be escaped. Numeric literals (vectors) never pass through here.
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

    ``score`` is ``None`` when not projected. The id column populates
    ``Document.id`` (not a metadata key); declared metadata columns and the
    JSON catch-all merge into ``Document.metadata``.
    """
    n = table.num_rows
    if n == 0:
        return []

    columns = {name: table.column(name).to_pylist() for name in table.column_names}
    ids = columns.get(id_column, [None] * n)
    texts = columns.get(text_column, [""] * n)
    metadata_json = columns.get(METADATA_JSON_COLUMN, [None] * n)
    scores = columns.get(SCORE_COLUMN, [None] * n)

    # Non-reserved columns are declared metadata to fold into Document.metadata.
    # "_id" is the engine's internal id (distinct from the user id_column).
    reserved = {id_column, text_column, METADATA_JSON_COLUMN, SCORE_COLUMN, "_id"}
    extra_columns = [name for name in columns if name not in reserved]

    results: list[tuple[Document, float | None]] = []
    for i in range(n):
        raw = metadata_json[i]
        metadata: dict[str, Any] = json.loads(raw) if raw else {}
        for name in extra_columns:
            metadata[name] = columns[name][i]
        doc = Document(id=ids[i], page_content=texts[i] or "", metadata=metadata)
        results.append((doc, scores[i]))
    return results
