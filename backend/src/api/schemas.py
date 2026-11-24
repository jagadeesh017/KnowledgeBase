from uuid import UUID

from pydantic import BaseModel, Field

from jobs.repository import JobStatus
from rag.schemas import Source


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    document_id: str | None = Field(default=None, max_length=200)
    top_k: int | None = Field(default=None, ge=1, le=50)


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


class IngestResponse(BaseModel):
    job_id: UUID
    document_id: str
    status: JobStatus = "pending"


class JobStatusResponse(BaseModel):
    job_id: UUID
    document_id: str
    filename: str
    status: JobStatus
    attempts: int
    max_attempts: int
    error: str | None
    chunk_count: int | None


class DeleteResponse(BaseModel):
    document_id: str
    deleted: bool


class HealthResponse(BaseModel):
    status: str
    collection: str
    embedding_model: str
    chat_model: str
