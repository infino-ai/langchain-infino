"""Unit tests for the Arrow/SQL bridge — pure functions, no engine needed."""

import json

import pyarrow as pa
import pytest
from langchain_core.documents import Document

from langchain_infino._arrow import (
    METADATA_JSON_COLUMN,
    SCORE_COLUMN,
    rows_to_documents,
    serialize_metadata,
    sql_lit,
    vector_array,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("abc", "'abc'"),
        ("a'b", "'a''b'"),
        ("''", "''''''"),
        ("doc-1", "'doc-1'"),
    ],
)
def test_sql_lit_escapes_quotes(value: str, expected: str) -> None:
    assert sql_lit(value) == expected


def test_serialize_metadata_is_deterministic() -> None:
    a = serialize_metadata({"b": 1, "a": 2})
    b = serialize_metadata({"a": 2, "b": 1})
    assert a == b == '{"a":2,"b":1}'


def test_vector_array_is_fixed_size_list() -> None:
    arr = vector_array([[1.0, 2.0], [3.0, 4.0]], dim=2)
    assert arr.type == pa.list_(pa.float32(), 2)
    assert arr.to_pylist() == [[1.0, 2.0], [3.0, 4.0]]


def test_rows_to_documents_merges_metadata_and_id() -> None:
    table = pa.table(
        {
            "doc_id": ["x1", "x2"],
            "page_content": ["hello", "world"],
            METADATA_JSON_COLUMN: [json.dumps({"src": "a"}), json.dumps({"src": "b"})],
            SCORE_COLUMN: [0.1, 0.2],
        }
    )
    results = rows_to_documents(table, id_column="doc_id", text_column="page_content")

    assert [score for _, score in results] == [0.1, 0.2]
    first = results[0][0]
    assert isinstance(first, Document)
    assert first.page_content == "hello"
    # The user id populates Document.id, not a metadata key.
    assert first.id == "x1"
    assert first.metadata == {"src": "a"}


def test_rows_to_documents_folds_declared_columns_and_skips_engine_id() -> None:
    table = pa.table(
        {
            "doc_id": ["a"],
            "page_content": ["hi"],
            "category": ["ml"],  # declared metadata column
            "_id": [99],  # engine-internal id — must not leak into metadata
            METADATA_JSON_COLUMN: [json.dumps({"extra": 1})],
            SCORE_COLUMN: [0.5],
        }
    )
    (doc, score), = rows_to_documents(
        table, id_column="doc_id", text_column="page_content"
    )
    assert doc.id == "a"
    assert score == 0.5
    # Declared column folds in; the catch-all merges; "_id" is dropped.
    assert doc.metadata == {"extra": 1, "category": "ml"}


def test_rows_to_documents_score_none_when_not_projected() -> None:
    table = pa.table(
        {"doc_id": ["a"], "page_content": ["hi"], METADATA_JSON_COLUMN: ["{}"]}
    )
    (doc, score), = rows_to_documents(
        table, id_column="doc_id", text_column="page_content"
    )
    assert score is None
    assert doc.metadata == {}


def test_rows_to_documents_empty() -> None:
    table = pa.table({"doc_id": pa.array([], type=pa.large_utf8())})
    assert (
        rows_to_documents(table, id_column="doc_id", text_column="page_content") == []
    )
