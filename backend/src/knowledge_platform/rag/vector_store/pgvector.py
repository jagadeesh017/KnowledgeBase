from typing import Any, ClassVar

from pgvector import Vector
from psycopg.types.json import Jsonb

from knowledge_platform.db.postgres import ConnectionProvider
from knowledge_platform.exceptions import ServiceUnavailableError
from knowledge_platform.rag.schemas import DocumentChunk, SearchResult
from knowledge_platform.rag.vector_store.base import VectorStore


class PgVectorStore(VectorStore):
    ALLOWED_COLUMNS: ClassVar[dict[str, str]] = {
        "collection": "collection",
        "tenant_id": "tenant_id",
        "document_id": "document_id",
        "ingest_version": "ingest_version",
    }

    def __init__(self, provider: ConnectionProvider) -> None:
        self.provider = provider

    def add(
        self,
        collection: str,
        chunks: list[DocumentChunk],
        vectors: list[list[float]],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return

        sql = """
            INSERT INTO document_chunks (
                collection, tenant_id, document_id, chunk_id, ingest_version,
                content, page_number, metadata, embedding
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
            ON CONFLICT (chunk_id) DO UPDATE SET
                content = EXCLUDED.content,
                page_number = EXCLUDED.page_number,
                metadata = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding,
                ingest_version = EXCLUDED.ingest_version
        """
        rows = [
            (
                collection,
                str(chunk.metadata.get("tenant_id", "")),
                chunk.document_id,
                chunk.chunk_id,
                str(chunk.metadata["ingest_version"]),
                chunk.content,
                chunk.page_number,
                Jsonb(chunk.metadata),
                Vector(vector),
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        try:
            with (
                self.provider.connection() as connection,
                connection.cursor() as cursor,
            ):
                cursor.executemany(sql, rows)
        except Exception as error:
            raise ServiceUnavailableError(
                f"Postgres write failed: {error}"
            ) from error

    def query(
        self,
        collection: str,
        vector: list[float],
        top_k: int = 5,
        where: dict[str, object] | None = None,
    ) -> list[SearchResult]:
        parameters: list[Any] = [Vector(vector)]
        where_sql = self._where_sql(collection, where, parameters)
        parameters.append(top_k)
        sql = f"""
            SELECT
                chunk_id,
                document_id,
                content,
                page_number,
                metadata,
                embedding <=> %s::vector AS distance
            FROM document_chunks
            WHERE {where_sql}
            ORDER BY distance
            LIMIT %s
        """
        try:
            with self.provider.connection() as connection:
                rows = connection.execute(sql, tuple(parameters)).fetchall()
        except Exception as error:
            raise ServiceUnavailableError(
                f"Postgres read failed: {error}"
            ) from error

        return [
            SearchResult(
                chunk_id=str(row[0]),
                document_id=str(row[1]),
                content=str(row[2]),
                page_number=int(row[3]),
                distance=float(row[5]),
                metadata={
                    str(key): str(value)
                    for key, value in dict(row[4]).items()
                    if key not in {"document_id", "page_number"}
                },
            )
            for row in rows
        ]

    def delete(self, collection: str, where: dict[str, object]) -> None:
        parameters: list[Any] = []
        where_sql = self._where_sql(collection, where, parameters)
        sql = f"DELETE FROM document_chunks WHERE {where_sql}"
        try:
            with self.provider.connection() as connection:
                connection.execute(sql, tuple(parameters))
        except Exception as error:
            raise ServiceUnavailableError(
                f"Postgres delete failed: {error}"
            ) from error

    def check_ready(self) -> None:
        try:
            with self.provider.connection() as connection:
                row = connection.execute("SELECT 1").fetchone()
        except Exception as error:
            raise ServiceUnavailableError(
                f"Postgres is unavailable: {error}"
            ) from error
        if row is None or row[0] != 1:
            raise ServiceUnavailableError("Postgres readiness check failed")

    def _where_sql(
        self,
        collection: str,
        where: dict[str, object] | None,
        parameters: list[Any],
    ) -> str:
        conditions = ["collection = %s"]
        parameters.append(collection)
        filter_sql = self._condition_sql(where, parameters)
        if filter_sql:
            conditions.append(filter_sql)
        return " AND ".join(conditions)

    def _condition_sql(
        self,
        where: dict[str, object] | None,
        parameters: list[Any],
    ) -> str:
        if not where:
            return ""

        if "$and" in where:
            children = where["$and"]
            if not isinstance(children, list):
                raise ValueError("$and must contain a list of filters")
            parts = []
            for child in children:
                if not isinstance(child, dict):
                    raise TypeError("$and children must be dictionaries")
                parts.append(self._condition_sql(child, parameters))
            return "(" + " AND ".join(parts) + ")"

        parts = []
        for key, value in where.items():
            column = self.ALLOWED_COLUMNS.get(key)
            if column is None:
                raise ValueError(f"Unsupported filter: {key}")
            if isinstance(value, dict) and "$ne" in value:
                parts.append(f"{column} <> %s")
                parameters.append(value["$ne"])
            else:
                parts.append(f"{column} = %s")
                parameters.append(value)
        return " AND ".join(parts)
