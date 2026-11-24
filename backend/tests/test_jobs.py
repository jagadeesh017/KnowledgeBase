from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import cast
from uuid import UUID, uuid4

from jobs.repository import IngestionJob, IngestionJobRepository
from jobs.worker import IngestionWorker
from rag.pipeline import RAGPipeline
from rag.schemas import DocumentChunk


class FakeResult:
    def __init__(self, rows: list[tuple[object, ...]]) -> None:
        self.rows = rows

    def fetchone(self) -> tuple[object, ...] | None:
        return self.rows[0] if self.rows else None


class FakeConnection:
    def __init__(self, rows: list[tuple[object, ...]]) -> None:
        self.rows = rows
        self.executed: list[tuple[str, tuple[object, ...]]] = []

    def execute(
        self, sql: str, parameters: tuple[object, ...] = ()
    ) -> FakeResult:
        self.executed.append((sql, parameters))
        return FakeResult(self.rows)


class FakeProvider:
    def __init__(self, rows: list[tuple[object, ...]] | None = None) -> None:
        self.connection_obj = FakeConnection(rows or [])

    @contextmanager
    def connection(self) -> Iterator[FakeConnection]:
        yield self.connection_obj


def job_row() -> tuple[object, ...]:
    return (
        uuid4(),
        "tenant_a",
        "doc_001",
        "guide.pdf",
        "processing",
        1,
        3,
        datetime.now(UTC),
        None,
        None,
    )


def test_claim_uses_skip_locked() -> None:
    rows = [job_row()]
    provider = FakeProvider(rows)
    repository = IngestionJobRepository(provider)

    job = repository.claim("worker-1")

    assert job is not None
    assert job.document_id == "doc_001"
    sql, parameters = provider.connection_obj.executed[-1]
    assert "FOR UPDATE SKIP LOCKED" in sql
    assert parameters == ("worker-1",)


def test_fail_schedules_retry_with_backoff() -> None:
    provider = FakeProvider()
    repository = IngestionJobRepository(provider, retry_backoff_seconds=5)
    job_id = uuid4()

    repository.fail(job_id, "boom", 1, 3)

    sql, parameters = provider.connection_obj.executed[-1]
    assert "status = 'pending'" in sql
    assert parameters == ("boom", 5.0, job_id)


def test_fail_marks_exhausted_job_failed() -> None:
    provider = FakeProvider()
    repository = IngestionJobRepository(provider, retry_backoff_seconds=5)
    job_id = uuid4()

    repository.fail(job_id, "boom", 3, 3)

    sql, parameters = provider.connection_obj.executed[-1]
    assert "status = 'failed'" in sql
    assert parameters == ("boom", job_id)


class FakeRepository:
    def __init__(self, job: IngestionJob) -> None:
        self.job = job
        self.claimed = False
        self.completed: list[tuple[UUID, int]] = []
        self.failed: list[tuple[UUID, str, int, int]] = []

    def claim(self, worker_id: str) -> IngestionJob | None:
        if self.claimed:
            return None
        self.claimed = True
        return self.job

    def fetch_source(self, job_id: UUID) -> bytes:
        return b"pdf"

    def complete(self, job_id: UUID, chunk_count: int) -> None:
        self.completed.append((job_id, chunk_count))

    def fail(
        self,
        job_id: UUID,
        error: str,
        attempts: int,
        max_attempts: int,
    ) -> None:
        self.failed.append((job_id, error, attempts, max_attempts))


class FakePipeline:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    def ingest_bytes(
        self,
        data: bytes,
        document_id: str,
        *,
        source: str = "document.pdf",
        tenant_id: str | None = None,
    ) -> list[DocumentChunk]:
        if self.error is not None:
            raise self.error
        return [
            DocumentChunk(
                chunk_id="chunk_1",
                document_id=document_id,
                content="hello",
                page_number=1,
            )
        ]


def make_job() -> IngestionJob:
    return IngestionJob(
        id=uuid4(),
        tenant_id="tenant_a",
        document_id="doc_001",
        filename="guide.pdf",
        status="processing",
        attempts=1,
        max_attempts=3,
        run_at=datetime.now(UTC),
        error=None,
        chunk_count=None,
    )


def test_worker_completes_job() -> None:
    job = make_job()
    repository = FakeRepository(job)
    worker = IngestionWorker(repository, cast(RAGPipeline, FakePipeline()))

    assert worker.run_once() is True

    assert repository.completed == [(job.id, 1)]
    assert repository.failed == []


def test_worker_fails_job_without_crashing() -> None:
    job = make_job()
    repository = FakeRepository(job)
    pipeline = FakePipeline(error=RuntimeError("ollama down"))
    worker = IngestionWorker(repository, cast(RAGPipeline, pipeline))

    assert worker.run_once() is True

    assert repository.completed == []
    assert repository.failed == [(job.id, "ollama down", 1, 3)]


def test_worker_returns_false_when_no_job_is_available() -> None:
    job = make_job()
    repository = FakeRepository(job)
    repository.claimed = True
    worker = IngestionWorker(repository, cast(RAGPipeline, FakePipeline()))

    assert worker.run_once() is False
