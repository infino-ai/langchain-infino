"""Unit tests for the structured-filter compiler and SQL helpers."""

import pyarrow as pa
import pytest

from langchain_infino._arrow import METADATA_JSON_COLUMN
from langchain_infino.vectorstores import (
    _build_schema,
    _compile_filter,
    _vector_literal,
)

ALLOWED = ["category", "year"]


def test_equality_filter() -> None:
    assert _compile_filter({"category": "physics"}, ALLOWED) == "category = 'physics'"


def test_multiple_clauses_are_anded() -> None:
    where = _compile_filter({"category": "ml", "year": 2024}, ALLOWED)
    assert where == "category = 'ml' AND year = 2024"


def test_operator_form() -> None:
    assert _compile_filter({"year": {"$gte": 2020}}, ALLOWED) == "year >= 2020"
    assert _compile_filter({"year": {"$lt": 2000}}, ALLOWED) == "year < 2000"


def test_in_operator() -> None:
    where = _compile_filter({"category": {"$in": ["ml", "physics"]}}, ALLOWED)
    assert where == "category IN ('ml', 'physics')"


def test_quotes_are_escaped() -> None:
    assert _compile_filter({"category": "a'b"}, ALLOWED) == "category = 'a''b'"


def test_undeclared_column_rejected() -> None:
    with pytest.raises(ValueError, match="not a declared metadata column"):
        _compile_filter({"unknown": 1}, ALLOWED)


def test_unsupported_operator_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported filter operator"):
        _compile_filter({"year": {"$regex": ".*"}}, ALLOWED)


def test_unsupported_value_type_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported filter value type"):
        _compile_filter({"category": [1, 2]}, ALLOWED)


def test_vector_literal() -> None:
    assert _vector_literal([1.0, 2.5, 3.0]) == "1.0,2.5,3.0"


def test_build_schema_with_metadata_columns() -> None:
    schema = _build_schema(
        dim=16,
        text_column="page_content",
        vector_column="embedding",
        id_column="doc_id",
        metadata_columns=[
            pa.field("category", pa.large_utf8()),
            pa.field("year", pa.int64()),
        ],
    )
    assert schema.names == [
        "doc_id",
        "page_content",
        "embedding",
        "category",
        "year",
        METADATA_JSON_COLUMN,
    ]
