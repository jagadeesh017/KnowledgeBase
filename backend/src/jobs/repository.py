from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, cast
from uuid import UUID, uuid4

from db.postgres import ConnectionProvider
from exceptions import ServiceUnavailableError

JobStatus = Literal["pending", "processing", "succeeded", "failed"]

JOB_COLUMNS = """
    id, tenant_id, document_id, filename, status, attempts, max_attempts,
    run_at, error, chunk_count
"""


@dataclass(frozen=True)
class IngestionJob:
    id: UUID
    tenant_id: str
    document_id: str
    filename: str
    status: JobStatus
    attempts: int
    max_attempts: int
    run_at: datetime
    error: str | None
    chunk_count: int | None


class IngestionJobRepository:
    def __init__(
        self,
        provider: ConnectionProvider,
        retry_backoff_seconds: float = 5.0,
    ) -> None:
        self.provider = provider
        self.retry_backoff_seconds = retry_backoff_seconds

    def create(
        self,
        source: bytes,
        *,
        tenant_id: str,
        document_id: str,
        filename: str,
        max_attempts: int,
    ) -> IngestionJob:
        job_id = uuid4()
        sql = f"""
            INSERT INTO ingestion_jobs (
                id, tenant_id, document_id, filename, source, max_attempts
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING {JOB_COLUMNS}
        """
        parameters = (
            job_id,
            tenant_id,
            document_id,
            filename,
            source,
            max_attempts,
        )
        try:
            with self.provider.connection() as connection:
                row = connection.execute(sql, parameters).fetchone()
        except Exception as error:
            raise ServiceUnavailableError(
                f"Postgres job write failed: {error}"
            ) from error
        if row is None:
            raise ServiceUnavailableError("Postgres did not return the created job")
        return self._to_job(row)

    def claim(self, worker_id: str) -> IngestionJob | None:
        sql = f"""
            WITH claimed AS (
                SELECT id
                FROM ingestion_jobs
                WHERE status IN ('pending', 'failed')
                  AND run_at <= now()
                  AND (locked_at IS NULL OR locked_at < now() - interval '5 minutes')
                ORDER BY run_at, created_at
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            UPDATE ingestion_jobs AS job
            SET status = 'processing',
                attempts = job.attempts + 1,
                locked_at = now(),
                locked_by = %s,
                updated_at = now()
            FROM claimed
            WHERE job.id = claimed.id
            RETURNING {JOB_COLUMNS.replace('id,', 'job.id,', 1)}
        """
        try:
            with self.provider.connection() as connection:
                row = connection.execute(sql, (worker_id,)).fetchone()
        except Exception as error:
            raise ServiceUnavailableError(
                f"Postgres job claim failed: {error}"
            ) from error
        return None if row is None else self._to_job(row)

    def fetch_source(self, job_id: UUID) -> bytes:
        sql = "SELECT source FROM ingestion_jobs WHERE id = %s"
        try:
            with self.provider.connection() as connection:
                row = connection.execute(sql, (job_id,)).fetchone()
        except Exception as error:
            raise ServiceUnavailableError(
                f"Postgres job read failed: {error}"
            ) from error
        if row is None:
            raise ServiceUnavailableError(f"Ingestion job {job_id} not found")
        return bytes(row[0])

    def complete(self, job_id: UUID, chunk_count: int) -> None:
        sql = """
            UPDATE ingestion_jobs
            SET status = 'succeeded',
                chunk_count = %s,
                error = NULL,
                locked_at = NULL,
                locked_by = NULL,
                updated_at = now()
            WHERE id = %s
        """
        try:
            with self.provider.connection() as connection:
                connection.execute(sql, (chunk_count, job_id))
        except Exception as error:
            raise ServiceUnavailableError(
                f"Postgres job completion failed: {error}"
            ) from error

    def fail(
        self,
        job_id: UUID,
        failure_reason: str,
        attempts: int,
        max_attempts: int,
    ) -> None:
        if attempts < max_attempts:
            delay = self.retry_backoff_seconds * (2 ** (attempts - 1))
            sql = """
                UPDATE ingestion_jobs
                SET status = 'pending',
                    error = %s,
                    run_at = now() + %s * interval '1 second',
                    locked_at = NULL,
                    locked_by = NULL,
                    updated_at = now()
                WHERE id = %s
            """
            parameters: tuple[Any, ...] = (failure_reason, delay, job_id)
        else:
            sql = """
                UPDATE ingestion_jobs
                SET status = 'failed',
                    error = %s,
                    locked_at = NULL,
                    locked_by = NULL,
                    updated_at = now()
                WHERE id = %s
            """
            parameters = (failure_reason, job_id)
        try:
            with self.provider.connection() as connection:
                connection.execute(sql, parameters)
        except Exception as error:
            raise ServiceUnavailableError(
                f"Postgres job failure update failed: {error}"
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

    def get(self, job_id: UUID, tenant_id: str) -> IngestionJob | None:
        sql = f"SELECT {JOB_COLUMNS} FROM ingestion_jobs WHERE id = %s AND tenant_id = %s"
        try:
            with self.provider.connection() as connection:
                row = connection.execute(sql, (job_id, tenant_id)).fetchone()
        except Exception as error:
            raise ServiceUnavailableError(
                f"Postgres job read failed: {error}"
            ) from error
        return None if row is None else self._to_job(row)

    @staticmethod
    def _to_job(row: Any) -> IngestionJob:
        return IngestionJob(
            id=UUID(str(row[0])),
            tenant_id=str(row[1]),
            document_id=str(row[2]),
            filename=str(row[3]),
            status=cast(JobStatus, str(row[4])),
            attempts=int(row[5]),
            max_attempts=int(row[6]),
            run_at=row[7],
            error=None if row[8] is None else str(row[8]),
            chunk_count=None if row[9] is None else int(row[9]),
        )
