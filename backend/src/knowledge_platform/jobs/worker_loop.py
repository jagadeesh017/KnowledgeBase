from knowledge_platform.config import get_settings
from knowledge_platform.container import (
    build_job_repository,
    build_pipeline,
    provider_from_settings,
)
from knowledge_platform.jobs.repository import IngestionJobRepository
from knowledge_platform.jobs.worker import IngestionWorker
from knowledge_platform.rag.pipeline import RAGPipeline


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
