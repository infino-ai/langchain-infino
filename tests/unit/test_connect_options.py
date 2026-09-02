"""``InfinoVectorStore.connect`` must forward exactly the options it is given.

Connection configuration is invisible after connect: a dropped keyword still
yields a working store, just not the configured one. Sending ``None`` for an
unset option is equally wrong — it overrides an engine default the caller
never expressed a preference about.
"""

from typing import Any
from unittest.mock import patch

import pyarrow as pa
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

import langchain_infino.vectorstores as vectorstores
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


class _StubConnection:
    """Stand-in for ``infino.Connection``, recording lifecycle calls.

    ``create_table`` rejects an existing name and ``list_tables`` reflects
    what was created, as the engine does.
    """

    def __init__(self, existing: tuple[str, ...] = ()) -> None:
        self.created_database = False
        self.created_table: tuple[str, Any] | None = None
        self.opened: list[str] = []
        self.tables = list(existing)

    def create_database(self) -> None:
        self.created_database = True

    def create_table(self, name: str, schema: Any, indexes: Any) -> object:
        if name in self.tables:
            raise ValueError(f"create_table({name}): {name}")
        self.tables.append(name)
        self.created_table = (name, indexes)
        return _StubTable()

    def open_table(self, name: str) -> object:
        if name not in self.tables:
            raise KeyError(name)
        self.opened.append(name)
        return _StubTable()

    def list_tables(self) -> list[str]:
        return list(self.tables)


@pytest.fixture
def connection() -> _StubConnection:
    """A connection whose table already exists — the plain open path."""
    return _StubConnection(existing=("docs",))


@pytest.fixture
def empty_connection() -> _StubConnection:
    """A connection with no tables yet — the creating path."""
    return _StubConnection()


def _connect(connection: _StubConnection, **kwargs: Any) -> dict[str, Any]:
    """Call ``connect`` with a stubbed engine, returning the forwarded kwargs."""
    with patch.object(
        vectorstores.infino, "connect", return_value=connection
    ) as spy:
        InfinoVectorStore.connect(
            "s3://bucket/prefix",
            DeterministicFakeEmbedding(size=EMBED_DIM),
            "docs",
            dim=EMBED_DIM,
            **kwargs,
        )
    return dict(spy.call_args.kwargs)


def test_unset_options_are_omitted_not_passed_as_none(connection) -> None:
    # Sending `validate=None` is not the same as not sending it: the engine
    # chooses its own default only for arguments it never receives.
    assert _connect(connection) == {}


def test_every_connection_option_reaches_the_engine(connection) -> None:
    forwarded = _connect(
        connection,
        storage_options={"aws_region": "us-east-1"},
        cache_dir="/tmp/infino-cache",
        cache_budget_bytes=64 << 20,
        connection_memory_budget_bytes=512 << 20,
        cold_fetch_mode="range_only",
        validate=True,
        api_key="secret",
    )
    assert forwarded == {
        "storage_options": {"aws_region": "us-east-1"},
        "cache_dir": "/tmp/infino-cache",
        "cache_budget_bytes": 64 << 20,
        "connection_memory_budget_bytes": 512 << 20,
        "cold_fetch_mode": "range_only",
        "validate": True,
        "api_key": "secret",
    }


def test_storage_options_are_copied_not_aliased(connection) -> None:
    # The caller's mapping must not stay live inside the connection.
    options = {"aws_region": "us-east-1"}
    forwarded = _connect(connection, storage_options=options)
    options["aws_region"] = "eu-west-1"
    assert forwarded["storage_options"] == {"aws_region": "us-east-1"}


def test_opens_an_existing_table_by_default(connection) -> None:
    _connect(connection)
    assert connection.opened == ["docs"]
    assert connection.created_table is None


def test_create_makes_the_table_when_it_is_absent(empty_connection) -> None:
    _connect(empty_connection, create=True)
    assert empty_connection.created_table is not None
    assert empty_connection.created_table[0] == "docs"
    # The handle from create_table is used directly, saving a redundant open.
    assert empty_connection.opened == []


def test_create_opens_the_table_when_it_is_already_there(connection) -> None:
    # `create=True` means "ensure it exists", so a second run must not fail on
    # the table its first run created.
    _connect(connection, create=True)
    assert connection.created_table is None
    assert connection.opened == ["docs"]


def test_database_is_provisioned_only_when_asked(connection) -> None:
    _connect(connection)
    assert connection.created_database is False


def test_create_database_provisions_before_the_table(empty_connection) -> None:
    _connect(empty_connection, create_database=True, create=True)
    assert empty_connection.created_database is True
    assert empty_connection.created_table is not None
