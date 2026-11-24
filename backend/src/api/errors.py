import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from exceptions import (
    KnowledgePlatformError,
    ServiceUnavailableError,
)

logger = logging.getLogger(__name__)


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(KnowledgePlatformError)
    async def handle_domain_error(
        request: Request, error: KnowledgePlatformError
    ) -> JSONResponse:
        logger.warning(
            "domain_error",
            extra={
                "path": request.url.path,
                "error_type": type(error).__name__,
                "detail": str(error),
            },
        )
        return JSONResponse(
            status_code=422,
            content={"detail": str(error), "error": type(error).__name__},
        )

    @app.exception_handler(ServiceUnavailableError)
    async def handle_unavailable_error(
        request: Request, error: ServiceUnavailableError
    ) -> JSONResponse:
        logger.warning(
            "dependency_unavailable",
            extra={
                "path": request.url.path,
                "error_type": type(error).__name__,
                "detail": str(error),
            },
        )
        return JSONResponse(
            status_code=503,
            content={"detail": str(error), "error": type(error).__name__},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, error: Exception) -> JSONResponse:
        logger.exception(
            "unexpected_error",
            extra={"path": request.url.path, "error_type": type(error).__name__},
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )
