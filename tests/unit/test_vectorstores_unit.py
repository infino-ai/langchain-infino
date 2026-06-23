"""Unit tests for store helpers that don't need a live connection."""

import pyarrow as pa
import pytest

from langchain_infino._arrow import METADATA_JSON_COLUMN
from langchain_infino.vectorstores import _RELEVANCE_FNS, _build_schema


def test_build_schema_shape() -> None:
    schema = _build_schema(
        dim=3, text_column="page_content", vector_column="embedding", id_column="doc_id"
    )
    assert schema.names == ["doc_id", "page_content", "embedding", METADATA_JSON_COLUMN]
    assert schema.field("embedding").type == pa.list_(pa.float32(), 3)
    assert schema.field("doc_id").type == pa.large_utf8()


def test_cosine_relevance_is_clamped_and_decreasing() -> None:
    fn = _RELEVANCE_FNS["cosine"]
    assert fn(0.0) == 1.0
    assert fn(1.0) == 0.0
    # Distances outside [0, 1] still clamp into the unit interval.
    assert fn(2.0) == 0.0
    assert fn(-0.5) == 1.0


def test_l2_relevance_is_monotone_decreasing() -> None:
    fn = _RELEVANCE_FNS["l2sq"]
    assert fn(0.0) == 1.0
    assert fn(1.0) == pytest.approx(0.5)
    assert fn(3.0) == pytest.approx(0.25)
    assert _RELEVANCE_FNS["l2"](0.0) == 1.0
