import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from knowledge_platform.api.dependencies import (
    TenantContext,
    get_job_repository,
    get_pipeline,
    get_settings,
    get_tenant,
)
from knowledge_platform.api.errors import register_error_handlers
from knowledge_platform.api.schemas import (
    AskRequest,
    AskResponse,
    DeleteResponse,
    HealthResponse,
    IngestResponse,
    JobStatusResponse,
)
from knowledge_platform.config import Settings
from knowledge_platform.config import get_settings as load_settings
from knowledge_platform.container import build_job_repository, build_pipeline
from knowledge_platform.exceptions import ServiceUnavailableError
from knowledge_platform.jobs.repository import IngestionJobRepository
from knowledge_platform.observability import (
    configure_logging,
    new_request_id,
    request_id_var,
)
from knowledge_platform.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


async def read_upload(upload: UploadFile, limit: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while chunk := await upload.read(1024 * 1024):
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code=413,
                detail=f"Uploaded file exceeds {limit} bytes",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def create_app(
    pipeline: RAGPipeline | None = None,
    settings: Settings | None = None,
    job_repository: IngestionJobRepository | None = None,
) -> FastAPI:
    active_settings = settings if settings is not None else load_settings()
    active_pipeline = pipeline
    active_job_repository = job_repository

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        nonlocal active_pipeline, active_job_repository
        configure_logging(active_settings.log_level)
        if active_pipeline is None:
            active_pipeline = build_pipeline(active_settings)
        if active_job_repository is None:
            active_job_repository = build_job_repository(active_settings)
        app.state.settings = active_settings
        app.state.pipeline = active_pipeline
        app.state.job_repository = active_job_repository
        logger.info(
            "service_started",
            extra={"collection": active_settings.default_collection},
        )
        yield

    app = FastAPI(
        title="Knowledge Platform API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = active_settings
    app.state.pipeline = active_pipeline
    app.state.job_repository = active_job_repository
    register_error_handlers(app)

    @app.middleware("http")
    async def request_context(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get("x-request-id") or new_request_id()
        token = request_id_var.set(request_id)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request_completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "request_id": request_id,
            },
        )
        return response

    @app.get("/health", response_model=HealthResponse)
    def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
        return HealthResponse(
            status="ok",
            collection=settings.default_collection,
            embedding_model=settings.embedding_model,
            chat_model=settings.chat_model,
        )

    @app.get("/ready")
    def ready(
        pipeline: RAGPipeline = Depends(get_pipeline),
        repository: IngestionJobRepository = Depends(get_job_repository),
    ) -> Response:
        try:
            pipeline.check_ready()
            repository.check_ready()
        except ServiceUnavailableError as error:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "detail": str(error)},
            )
        return JSONResponse(content={"status": "ready"})

    @app.post("/documents", response_model=IngestResponse, status_code=202)
    async def ingest_document(
        file: UploadFile = File(...),
        document_id: str | None = Form(default=None, max_length=200),
        settings: Settings = Depends(get_settings),
        repository: IngestionJobRepository = Depends(get_job_repository),
        tenant: TenantContext = Depends(get_tenant),
    ) -> IngestResponse:
        filename = file.filename or "document.pdf"
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        data = await read_upload(file, settings.max_upload_bytes)
        if not data:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        resolved_document_id = document_id or uuid4().hex
        job = repository.create(
            data,
            tenant_id=tenant.tenant_id or "",
            document_id=resolved_document_id,
            filename=filename,
            max_attempts=settings.job_max_attempts,
        )
        return IngestResponse(job_id=job.id, document_id=job.document_id)

    @app.get("/jobs/{job_id}", response_model=JobStatusResponse)
    def get_job(
        job_id: UUID,
        repository: IngestionJobRepository = Depends(get_job_repository),
        tenant: TenantContext = Depends(get_tenant),
    ) -> JobStatusResponse:
        job = repository.get(job_id, tenant.tenant_id or "")
        if job is None:
            raise HTTPException(status_code=404, detail="Ingestion job not found")
        return JobStatusResponse(
            job_id=job.id,
            document_id=job.document_id,
            filename=job.filename,
            status=job.status,
            attempts=job.attempts,
            max_attempts=job.max_attempts,
            error=job.error,
            chunk_count=job.chunk_count,
        )

    @app.post("/ask", response_model=AskResponse)
    def ask(
        payload: AskRequest,
        pipeline: RAGPipeline = Depends(get_pipeline),
        settings: Settings = Depends(get_settings),
        tenant: TenantContext = Depends(get_tenant),
    ) -> AskResponse:
        where: dict[str, object] | None = (
            {"document_id": payload.document_id} if payload.document_id else None
        )
        answer = pipeline.answer(
            payload.question,
            top_k=payload.top_k if payload.top_k is not None else settings.top_k,
            where=where,
            tenant_id=tenant.tenant_id,
        )
        return AskResponse(answer=answer.answer, sources=answer.sources)

    @app.delete("/documents/{document_id}", response_model=DeleteResponse)
    def delete_document(
        document_id: str,
        pipeline: RAGPipeline = Depends(get_pipeline),
        tenant: TenantContext = Depends(get_tenant),
    ) -> DeleteResponse:
        pipeline.delete_document(document_id, tenant_id=tenant.tenant_id)
        return DeleteResponse(document_id=document_id, deleted=True)

    return app


app = create_app()
