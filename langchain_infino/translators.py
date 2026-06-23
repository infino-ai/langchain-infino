"""Self-query translator: lower a LangChain ``StructuredQuery`` to an Infino
metadata filter.

Plugs into ``SelfQueryRetriever``. Infino evaluates the filter as a SQL
``WHERE``, so the full comparison and boolean surface is available, not a
reduced DSL.
"""

from __future__ import annotations

from typing import Any

from langchain_core.structured_query import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
)

# LangChain comparators / operators → the keys InfinoVectorStore's filter
# compiler understands.
_COMPARATOR_TO_OP = {
    Comparator.EQ: "$eq",
    Comparator.NE: "$ne",
    Comparator.GT: "$gt",
    Comparator.GTE: "$gte",
    Comparator.LT: "$lt",
    Comparator.LTE: "$lte",
    Comparator.IN: "$in",
    Comparator.NIN: "$nin",
}
_OPERATOR_TO_KEY = {
    Operator.AND: "$and",
    Operator.OR: "$or",
    Operator.NOT: "$not",
}


class InfinoTranslator(Visitor):
    """Translate a ``StructuredQuery`` into an Infino metadata filter."""

    allowed_comparators = list(_COMPARATOR_TO_OP)
    allowed_operators = list(_OPERATOR_TO_KEY)

    def visit_comparison(self, comparison: Comparison) -> dict[str, Any]:
        op = _COMPARATOR_TO_OP[comparison.comparator]
        return {comparison.attribute: {op: comparison.value}}

    def visit_operation(self, operation: Operation) -> dict[str, Any]:
        arguments = [arg.accept(self) for arg in operation.arguments]
        return {_OPERATOR_TO_KEY[operation.operator]: arguments}

    def visit_structured_query(
        self, structured_query: StructuredQuery
    ) -> tuple[str, dict[str, Any]]:
        if structured_query.filter is None:
            kwargs: dict[str, Any] = {}
        else:
            kwargs = {"filter": structured_query.filter.accept(self)}
        return structured_query.query, kwargs
