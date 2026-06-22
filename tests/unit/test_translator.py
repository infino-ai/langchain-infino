"""Unit tests for the self-query translator and the logical-operator
extensions to the filter compiler."""

import pytest
from langchain_core.structured_query import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
)

from langchain_infino.translators import InfinoTranslator
from langchain_infino.vectorstores import _compile_filter

ALLOWED = ["category", "year"]


def test_visit_comparison() -> None:
    t = InfinoTranslator()
    out = t.visit_comparison(
        Comparison(comparator=Comparator.GTE, attribute="year", value=2020)
    )
    assert out == {"year": {"$gte": 2020}}


@pytest.mark.parametrize(
    ("comparator", "op"),
    [
        (Comparator.EQ, "$eq"),
        (Comparator.NE, "$ne"),
        (Comparator.GT, "$gt"),
        (Comparator.GTE, "$gte"),
        (Comparator.LT, "$lt"),
        (Comparator.LTE, "$lte"),
        (Comparator.IN, "$in"),
        (Comparator.NIN, "$nin"),
    ],
)
def test_every_comparator_maps(comparator: Comparator, op: str) -> None:
    out = InfinoTranslator().visit_comparison(
        Comparison(comparator=comparator, attribute="x", value=1)
    )
    assert out == {"x": {op: 1}}


@pytest.mark.parametrize(
    ("operator", "key"),
    [(Operator.AND, "$and"), (Operator.OR, "$or"), (Operator.NOT, "$not")],
)
def test_every_operator_maps(operator: Operator, key: str) -> None:
    operation = Operation(
        operator=operator,
        arguments=[Comparison(comparator=Comparator.EQ, attribute="x", value=1)],
    )
    assert key in InfinoTranslator().visit_operation(operation)


def test_visit_operation_and() -> None:
    t = InfinoTranslator()
    op = Operation(
        operator=Operator.AND,
        arguments=[
            Comparison(comparator=Comparator.EQ, attribute="category", value="ml"),
            Comparison(comparator=Comparator.GTE, attribute="year", value=2020),
        ],
    )
    assert t.visit_operation(op) == {
        "$and": [{"category": {"$eq": "ml"}}, {"year": {"$gte": 2020}}]
    }


def test_visit_structured_query() -> None:
    t = InfinoTranslator()
    sq = StructuredQuery(
        query="neural nets",
        filter=Comparison(comparator=Comparator.EQ, attribute="category", value="ml"),
        limit=None,
    )
    query, kwargs = t.visit_structured_query(sq)
    assert query == "neural nets"
    assert kwargs == {"filter": {"category": {"$eq": "ml"}}}


def test_structured_query_without_filter() -> None:
    t = InfinoTranslator()
    query, kwargs = t.visit_structured_query(
        StructuredQuery(query="anything", filter=None, limit=None)
    )
    assert query == "anything"
    assert kwargs == {}


# The translator's output must compile to valid SQL.


def test_compile_and() -> None:
    where = _compile_filter(
        {"$and": [{"category": "ml"}, {"year": {"$gte": 2020}}]}, ALLOWED
    )
    assert where == "((category = 'ml') AND (year >= 2020))"


def test_compile_or() -> None:
    where = _compile_filter(
        {"$or": [{"category": "ml"}, {"category": "physics"}]}, ALLOWED
    )
    assert where == "((category = 'ml') OR (category = 'physics'))"


def test_compile_not() -> None:
    where = _compile_filter({"$not": [{"category": "ml"}]}, ALLOWED)
    assert where == "NOT (category = 'ml')"


def test_compile_not_accepts_a_bare_dict() -> None:
    # $not also accepts a mapping directly, not only a single-element list.
    where = _compile_filter({"$not": {"category": "ml"}}, ALLOWED)
    assert where == "NOT (category = 'ml')"


def test_compile_nin() -> None:
    where = _compile_filter({"category": {"$nin": ["ml", "physics"]}}, ALLOWED)
    assert where == "category NOT IN ('ml', 'physics')"
