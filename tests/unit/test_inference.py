"""What the store infers when the caller doesn't spell it out.

Promotion decides what is filterable, and it is decided once at table
creation — so a key wrongly promoted (or wrongly skipped) is not something a
later call can correct.
"""

import pyarrow as pa
import pytest

from langchain_infino.vectorstores import (
    _dim_from_schema,
    _infer_metadata_columns,
    _reserved_columns,
)

RESERVED = _reserved_columns("page_content", "embedding", "doc_id")


def _infer(metadatas: list[dict]) -> dict[str, pa.DataType]:
    return {f.name: f.type for f in _infer_metadata_columns(metadatas, RESERVED)}


def test_scalar_keys_map_to_their_arrow_types() -> None:
    assert _infer([{"s": "x", "i": 1, "f": 1.5, "b": True}]) == {
        "s": pa.large_utf8(),
        "i": pa.int64(),
        "f": pa.float64(),
        "b": pa.bool_(),
    }


def test_bool_is_not_mistaken_for_int() -> None:
    # bool subclasses int in Python, so order of checks matters.
    assert _infer([{"b": False}]) == {"b": pa.bool_()}


def test_int_and_float_together_widen_to_float() -> None:
    assert _infer([{"n": 1}, {"n": 2.5}]) == {"n": pa.float64()}


def test_conflicting_types_stay_in_the_json_catch_all() -> None:
    assert _infer([{"x": "a"}, {"x": 1}]) == {}


def test_nested_values_stay_in_the_json_catch_all() -> None:
    assert _infer([{"tags": ["a"], "nested": {"k": 1}}]) == {}


def test_a_key_that_is_only_ever_null_is_not_promoted() -> None:
    # There is no type to give the column.
    assert _infer([{"x": None}, {"x": None}]) == {}


def test_nulls_alongside_values_still_promote() -> None:
    assert _infer([{"x": None}, {"x": "a"}]) == {"x": pa.large_utf8()}


def test_promoted_columns_are_nullable() -> None:
    # The schema is fixed at creation, so a later append may omit the key.
    fields = _infer_metadata_columns([{"x": "a"}], RESERVED)
    assert all(f.nullable for f in fields)


@pytest.mark.parametrize("name", sorted(RESERVED))
def test_structural_column_names_are_never_promoted(name: str) -> None:
    # Promoting one would collide with a column the engine adds itself, which
    # surfaces as a duplicate-column error mid-query.
    assert _infer([{name: "x"}]) == {}


def test_a_sparse_key_is_promoted_from_whichever_document_has_it() -> None:
    assert _infer([{"a": 1}, {"b": "x"}]) == {"a": pa.int64(), "b": pa.large_utf8()}


def test_dim_is_read_off_the_vector_column() -> None:
    schema = pa.schema([pa.field("embedding", pa.list_(pa.float32(), 384))])
    assert _dim_from_schema(schema, "embedding") == 384


def test_dim_from_a_missing_vector_column_says_which_columns_exist() -> None:
    schema = pa.schema([pa.field("other", pa.large_utf8())])
    with pytest.raises(ValueError, match="no vector column"):
        _dim_from_schema(schema, "embedding")


def test_dim_from_a_variable_length_column_asks_for_an_explicit_dim() -> None:
    schema = pa.schema([pa.field("embedding", pa.list_(pa.float32()))])
    with pytest.raises(ValueError, match="not a fixed-size list"):
        _dim_from_schema(schema, "embedding")
