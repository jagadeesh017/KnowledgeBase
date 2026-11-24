from config import get_settings
from container import (
    build_job_repository,
    build_pipeline,
    provider_from_settings,
)
from jobs.repository import IngestionJobRepository
from jobs.worker import IngestionWorker
from rag.pipeline import RAGPipeline


def run(repository: IngestionJobRepository, pipeline: RAGPipeline) -> None:
    settings = get_settings()
    worker = IngestionWorker(repository, pipeline)
    worker.run_forever(settings.job_poll_interval_seconds)


def main() -> None:
    settings = get_settings()
    provider = provider_from_settings(settings)
    repository = build_job_repository(settings, provider)
    pipeline = build_pipeline(settings, provider)
    run(repository, pipeline)


if __name__ == "__main__":
    main()
