"""``open_or_create`` must be idempotent without hiding real failures.

The existence check and the create are separate calls, so another writer can
win the name in between. Tolerating that is the point — but narrowly:
``create_table`` also rejects a table it cannot build, and swallowing that
would surface as a confusing error on the first write instead.
"""

from typing import Any

import pyarrow as pa
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from langchain_infino import InfinoVectorStore

EMBED_DIM = 16

# The schema a store reads back to learn its embedding width and which
# metadata keys were promoted.
def _stub_schema(dim: int = EMBED_DIM) -> pa.Schema:
    return pa.schema(
        [
            pa.field("doc_id", pa.large_utf8(), nullable=False),
            pa.field("page_content", pa.large_utf8(), nullable=False),
            pa.field("embedding", pa.list_(pa.float32(), dim), nullable=False),
            pa.field("_metadata_json", pa.large_utf8(), nullable=False),
        ]
    )


class _StubTable:
    def schema(self) -> pa.Schema:
        return _stub_schema()


class _RacingConnection:
    """A connection losing the creation race.

    ``list_tables`` reports nothing on the first call, ``create_table`` then
    fails as though another writer got there first, and ``appears`` decides
    whether the table is visible afterwards.
    """

    def __init__(self, *, appears: bool) -> None:
        self._appears = appears
        self._checks = 0
        self.opened: list[str] = []

    def list_tables(self) -> list[str]:
        self._checks += 1
        if self._checks == 1:
            return []
        return ["docs"] if self._appears else []

    def create_table(self, name: str, schema: Any, indexes: Any) -> object:
        raise ValueError(f"create_table({name}): {name}")

    def open_table(self, name: str) -> object:
        self.opened.append(name)
        return _StubTable()


def _open_or_create(connection: Any) -> InfinoVectorStore:
    return InfinoVectorStore.open_or_create(
        connection,
        "docs",
        DeterministicFakeEmbedding(size=EMBED_DIM),
        dim=EMBED_DIM,
    )


def test_a_lost_creation_race_falls_back_to_opening() -> None:
    connection = _RacingConnection(appears=True)
    store = _open_or_create(connection)
    assert connection.opened == ["docs"]
    assert store.table_name == "docs"


def test_a_genuine_creation_failure_still_raises() -> None:
    # No table appeared, so the create failed on its own merits.
    connection = _RacingConnection(appears=False)
    with pytest.raises(ValueError, match="create_table"):
        _open_or_create(connection)
