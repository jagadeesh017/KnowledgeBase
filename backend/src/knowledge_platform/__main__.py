import uvicorn

from knowledge_platform.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "knowledge_platform.api.app:app",
        host="0.0.0.0",
        port=8000,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
