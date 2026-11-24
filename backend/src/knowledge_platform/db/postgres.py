from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Any, Protocol

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

MIGRATIONS_PATH = Path(__file__).resolve().parents[3] / "migrations"


def configure_connection(connection: psycopg.Connection[Any]) -> None:
    register_vector(connection)


class ConnectionProvider(Protocol):
    def connection(self) -> AbstractContextManager[psycopg.Connection[Any]]: ...


class PsycopgConnectionProvider:
    def __init__(
        self,
        dsn: str,
        min_size: int = 1,
        max_size: int = 10,
    ) -> None:
        self.pool = ConnectionPool(
            conninfo=dsn,
            min_size=min_size,
            max_size=max_size,
            open=True,
            configure=configure_connection,
        )

    @contextmanager
    def connection(self) -> Iterator[psycopg.Connection[Any]]:
        with self.pool.connection() as connection:
            yield connection


def apply_migrations(provider: ConnectionProvider) -> None:
    for path in sorted(MIGRATIONS_PATH.glob("*.sql")):
        with provider.connection() as connection:
            connection.execute(path.read_text())
