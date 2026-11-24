import secrets

from fastapi import Header, HTTPException, Request
from pydantic import BaseModel

from config import Settings
from jobs.repository import IngestionJobRepository
from rag.pipeline import RAGPipeline


class TenantContext(BaseModel):
    tenant_id: str | None


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_pipeline(request: Request) -> RAGPipeline:
    return request.app.state.pipeline


def get_job_repository(request: Request) -> IngestionJobRepository:
    return request.app.state.job_repository


def get_tenant(
    request: Request,
    authorization: str | None = Header(default=None),
) -> TenantContext:
    settings = get_settings(request)
    if not settings.auth_enabled:
        return TenantContext(tenant_id=None)

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = authorization.removeprefix("Bearer ").strip()
    for api_key, tenant_id in settings.api_keys.items():
        if secrets.compare_digest(api_key, token):
            return TenantContext(tenant_id=tenant_id)

    raise HTTPException(
        status_code=401,
        detail="Missing or invalid API key",
        headers={"WWW-Authenticate": "Bearer"},
    )
