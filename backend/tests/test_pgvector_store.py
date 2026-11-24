from collections.abc import Iterator
from contextlib import contextmanager
from typing import Self
from uuid import uuid4

import pytest
from pgvector import Vector

from rag.schemas import DocumentChunk
from rag.vector_store import PgVectorStore


class FakeCursor:
    def __init__(self) -> None:
        self.rows: list[tuple[object, ...]] = []
        self.executed: list[tuple[str, tuple[object, ...]]] = []

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def executemany(self, sql: str, rows: list[tuple[object, ...]]) -> None:
        self.executed.append((sql, ()))
        self.rows = rows

    def execute(self, sql: str, parameters: tuple[object, ...] = ()) -> "FakeCursor":
        self.executed.append((sql, parameters))
        return self

    def fetchall(self) -> list[tuple[object, ...]]:
        return self.rows

    def fetchone(self) -> tuple[object, ...] | None:
        return self.rows[0] if self.rows else None


class FakeConnection:
    def __init__(self) -> None:
        self.cursor_obj = FakeCursor()
        self.executed: list[tuple[str, tuple[object, ...]]] = []

    def cursor(self) -> FakeCursor:
        return self.cursor_obj

    def execute(self, sql: str, parameters: tuple[object, ...] = ()) -> FakeCursor:
        self.executed.append((sql, parameters))
        return self.cursor_obj


class FakeProvider:
    def __init__(self) -> None:
        self.connection_obj = FakeConnection()

    @contextmanager
    def connection(self) -> Iterator[FakeConnection]:
        yield self.connection_obj


def chunk() -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"{uuid4()}",
        document_id="doc_001",
        content="hello",
        page_number=1,
        metadata={"tenant_id": "tenant_a", "ingest_version": "v1"},
    )


def test_add_writes_pgvector_rows() -> None:
    provider = FakeProvider()
    store = PgVectorStore(provider)

    store.add("knowledge", [chunk()], [[0.1, 0.2]])

    row = provider.connection_obj.cursor_obj.rows[0]
    assert row[0] == "knowledge"
    assert row[1] == "tenant_a"
    assert isinstance(row[8], Vector)


def test_query_builds_tenant_and_version_filters() -> None:
    provider = FakeProvider()
    provider.connection_obj.cursor_obj.rows = [
        ("chunk_1", "doc_001", "hello", 1, {"source": "a.pdf"}, 0.25)
    ]
    store = PgVectorStore(provider)

    results = store.query(
        "knowledge",
        [0.1, 0.2],
        where={
            "$and": [
                {"tenant_id": "tenant_a"},
                {"ingest_version": {"$ne": "v1"}},
            ]
        },
    )

    sql, parameters = provider.connection_obj.executed[-1]
    assert "collection = %s" in sql
    assert "tenant_id = %s" in sql
    assert "ingest_version <> %s" in sql
    assert parameters == (Vector([0.1, 0.2]), "knowledge", "tenant_a", "v1", 5)
    assert results[0].distance == 0.25
    assert results[0].metadata == {"source": "a.pdf"}


def test_query_rejects_unknown_filter() -> None:
    store = PgVectorStore(FakeProvider())

    with pytest.raises(ValueError, match="Unsupported filter"):
        store.query("knowledge", [0.1], where={"secret": "value"})


def test_check_ready() -> None:
    provider = FakeProvider()
    provider.connection_obj.cursor_obj.rows = [(1,)]
    store = PgVectorStore(provider)

    store.check_ready()
