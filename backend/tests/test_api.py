from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import cast
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from api.app import create_app
from config import Settings
from exceptions import ServiceUnavailableError
from jobs.repository import IngestionJob, IngestionJobRepository
from rag.pipeline import RAGPipeline
from rag.schemas import DocumentChunk, RAGAnswer, Source


class FakePipeline:
    def __init__(self) -> None:
        self.ingested: list[tuple[bytes, str, str, str | None]] = []
        self.asked: list[
            tuple[str, int, dict[str, object] | None, str | None]
        ] = []
        self.deleted: list[tuple[str, str | None]] = []
        self.ingest_error: Exception | None = None
        self.ready_error: Exception | None = None

    def ingest_bytes(
        self,
        data: bytes,
        document_id: str,
        *,
        source: str = "document.pdf",
        tenant_id: str | None = None,
    ) -> list[DocumentChunk]:
        if self.ingest_error is not None:
            raise self.ingest_error
        self.ingested.append((data, document_id, source, tenant_id))
        return [
            DocumentChunk(
                chunk_id=f"{document_id}:chunk_001",
                document_id=document_id,
                content="RAG stores document chunks as vectors.",
                page_number=1,
                metadata={"source": source},
            )
        ]

    def answer(
        self,
        question: str,
        top_k: int = 5,
        where: dict[str, object] | None = None,
        *,
        tenant_id: str | None = None,
    ) -> RAGAnswer:
        self.asked.append((question, top_k, where, tenant_id))
        return RAGAnswer(
            answer="stub answer [1]",
            sources=[
                Source(
                    index=1,
                    chunk_id="doc_001:chunk_001",
                    document_id="doc_001",
                    source="guide.pdf",
                    page_number=1,
                    distance=0.1,
                )
            ],
        )

    def delete_document(
        self,
        document_id: str,
        *,
        tenant_id: str | None = None,
    ) -> None:
        self.deleted.append((document_id, tenant_id))

    def check_ready(self) -> None:
        if self.ready_error is not None:
            raise self.ready_error


class FakeJobRepository:
    def __init__(self) -> None:
        self.jobs: dict[UUID, IngestionJob] = {}
        self.sources: dict[UUID, bytes] = {}
        self.create_error: Exception | None = None
        self.ready_error: Exception | None = None

    def create(
        self,
        source: bytes,
        *,
        tenant_id: str,
        document_id: str,
        filename: str,
        max_attempts: int,
    ) -> IngestionJob:
        if self.create_error is not None:
            raise self.create_error
        job = IngestionJob(
            id=uuid4(),
            tenant_id=tenant_id,
            document_id=document_id,
            filename=filename,
            status="pending",
            attempts=0,
            max_attempts=max_attempts,
            run_at=datetime.now(UTC),
            error=None,
            chunk_count=None,
        )
        self.jobs[job.id] = job
        self.sources[job.id] = source
        return job

    def check_ready(self) -> None:
        if self.ready_error is not None:
            raise self.ready_error

    def get(self, job_id: UUID, tenant_id: str) -> IngestionJob | None:
        job = self.jobs.get(job_id)
        if job is None or job.tenant_id != tenant_id:
            return None
        return job


@pytest.fixture
def fake_pipeline() -> FakePipeline:
    return FakePipeline()


@pytest.fixture
def fake_job_repository() -> FakeJobRepository:
    return FakeJobRepository()


@pytest.fixture
def client(
    fake_pipeline: FakePipeline,
    fake_job_repository: FakeJobRepository,
    tmp_path: Path,
) -> TestClient:
    settings = Settings(
        chroma_path=tmp_path / "chroma",
        auth_enabled=False,
    )
    app = create_app(
        pipeline=cast(RAGPipeline, fake_pipeline),
        settings=settings,
        job_repository=cast(IngestionJobRepository, fake_job_repository),
    )
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_client(
    fake_pipeline: FakePipeline,
    fake_job_repository: FakeJobRepository,
    tmp_path: Path,
) -> TestClient:
    settings = Settings(
        chroma_path=tmp_path / "chroma",
        auth_enabled=True,
        api_keys={"secret-key": "tenant_a"},
    )
    app = create_app(
        pipeline=cast(RAGPipeline, fake_pipeline),
        settings=settings,
        job_repository=cast(IngestionJobRepository, fake_job_repository),
    )
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def limited_client(
    fake_pipeline: FakePipeline,
    fake_job_repository: FakeJobRepository,
    tmp_path: Path,
) -> TestClient:
    settings = Settings(
        chroma_path=tmp_path / "chroma",
        auth_enabled=False,
        max_upload_bytes=16,
    )
    app = create_app(
        pipeline=cast(RAGPipeline, fake_pipeline),
        settings=settings,
        job_repository=cast(IngestionJobRepository, fake_job_repository),
    )
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def pdf_bytes() -> bytes:
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.drawString(72, 750, "RAG stores document chunks as vectors.")
    pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def test_health(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.headers["x-request-id"]


def test_upload_pdf_enqueues_job(
    client: TestClient,
    fake_job_repository: FakeJobRepository,
    pdf_bytes: bytes,
) -> None:
    response = client.post(
        "/documents",
        files={"file": ("guide.pdf", pdf_bytes, "application/pdf")},
        data={"document_id": "doc_001"},
    )

    assert response.status_code == 202
    assert response.json()["document_id"] == "doc_001"
    assert response.json()["status"] == "pending"
    job_id = UUID(response.json()["job_id"])
    job = fake_job_repository.jobs[job_id]
    assert job.document_id == "doc_001"
    assert job.filename == "guide.pdf"
    assert fake_job_repository.sources[job_id] == pdf_bytes


def test_upload_rejects_non_pdf(client: TestClient) -> None:
    response = client.post(
        "/documents",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only PDF files are supported"


def test_ask_passes_filter_and_top_k(
    client: TestClient, fake_pipeline: FakePipeline
) -> None:
    response = client.post(
        "/ask",
        json={"question": "What does RAG store?", "document_id": "doc_001", "top_k": 3},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "stub answer [1]"
    assert response.json()["sources"][0]["page_number"] == 1
    assert fake_pipeline.asked == [
        ("What does RAG store?", 3, {"document_id": "doc_001"}, None)
    ]


def test_ask_uses_default_top_k(
    client: TestClient, fake_pipeline: FakePipeline
) -> None:
    response = client.post("/ask", json={"question": "What does RAG store?"})

    assert response.status_code == 200
    assert fake_pipeline.asked == [("What does RAG store?", 5, None, None)]


def test_delete_document(client: TestClient, fake_pipeline: FakePipeline) -> None:
    response = client.delete("/documents/doc_001")

    assert response.status_code == 200
    assert response.json() == {"document_id": "doc_001", "deleted": True}
    assert fake_pipeline.deleted == [("doc_001", None)]


def test_job_repository_error_becomes_503(
    client: TestClient,
    fake_job_repository: FakeJobRepository,
    pdf_bytes: bytes,
) -> None:
    fake_job_repository.create_error = ServiceUnavailableError("postgres down")

    response = client.post(
        "/documents",
        files={"file": ("guide.pdf", pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 503
    assert response.json()["error"] == "ServiceUnavailableError"


def test_custom_request_id_is_echoed(client: TestClient) -> None:
    request_id = uuid4().hex

    response = client.get("/health", headers={"x-request-id": request_id})

    assert response.headers["x-request-id"] == request_id


def test_health_does_not_require_auth(auth_client: TestClient) -> None:
    response = auth_client.get("/health")

    assert response.status_code == 200


def test_ask_requires_api_key(auth_client: TestClient) -> None:
    response = auth_client.post("/ask", json={"question": "hello"})

    assert response.status_code == 401
    assert response.json()["detail"] == "Missing or invalid API key"


def test_ask_rejects_wrong_api_key(auth_client: TestClient) -> None:
    response = auth_client.post(
        "/ask",
        json={"question": "hello"},
        headers={"Authorization": "Bearer wrong-key"},
    )

    assert response.status_code == 401


def test_ask_passes_tenant_from_api_key(
    auth_client: TestClient, fake_pipeline: FakePipeline
) -> None:
    response = auth_client.post(
        "/ask",
        json={"question": "hello"},
        headers={"Authorization": "Bearer secret-key"},
    )

    assert response.status_code == 200
    assert fake_pipeline.asked == [("hello", 5, None, "tenant_a")]


def test_upload_passes_tenant_from_api_key(
    auth_client: TestClient,
    fake_job_repository: FakeJobRepository,
    pdf_bytes: bytes,
) -> None:
    response = auth_client.post(
        "/documents",
        files={"file": ("guide.pdf", pdf_bytes, "application/pdf")},
        headers={"Authorization": "Bearer secret-key"},
    )

    assert response.status_code == 202
    job = fake_job_repository.jobs[UUID(response.json()["job_id"])]
    assert job.tenant_id == "tenant_a"


def test_delete_passes_tenant_from_api_key(
    auth_client: TestClient, fake_pipeline: FakePipeline
) -> None:
    response = auth_client.delete(
        "/documents/doc_001",
        headers={"Authorization": "Bearer secret-key"},
    )

    assert response.status_code == 200
    assert fake_pipeline.deleted == [("doc_001", "tenant_a")]


def test_ready(client: TestClient) -> None:
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_ready_returns_503_when_dependency_is_down(
    client: TestClient, fake_pipeline: FakePipeline
) -> None:
    fake_pipeline.ready_error = ServiceUnavailableError("ollama down")

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "detail": "ollama down",
    }


def test_upload_rejects_oversized_pdf(
    limited_client: TestClient, pdf_bytes: bytes
) -> None:
    response = limited_client.post(
        "/documents",
        files={"file": ("guide.pdf", pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "Uploaded file exceeds 16 bytes"


def test_job_status(client: TestClient, pdf_bytes: bytes) -> None:
    upload = client.post(
        "/documents",
        files={"file": ("guide.pdf", pdf_bytes, "application/pdf")},
        data={"document_id": "doc_001"},
    )
    job_id = upload.json()["job_id"]

    response = client.get(f"/jobs/{job_id}")

    assert response.status_code == 200
    assert response.json() == {
        "job_id": job_id,
        "document_id": "doc_001",
        "filename": "guide.pdf",
        "status": "pending",
        "attempts": 0,
        "max_attempts": 3,
        "error": None,
        "chunk_count": None,
    }


def test_job_status_requires_auth(auth_client: TestClient) -> None:
    response = auth_client.get(f"/jobs/{uuid4()}")

    assert response.status_code == 401
