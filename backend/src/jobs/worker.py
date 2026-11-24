import logging
import time
from uuid import uuid4

from jobs.repository import IngestionJobRepository
from rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class IngestionWorker:
    def __init__(
        self,
        repository: IngestionJobRepository,
        pipeline: RAGPipeline,
        worker_id: str | None = None,
    ) -> None:
        self.repository = repository
        self.pipeline = pipeline
        self.worker_id = worker_id or uuid4().hex

    def run_once(self) -> bool:
        job = self.repository.claim(self.worker_id)
        if job is None:
            return False

        try:
            source = self.repository.fetch_source(job.id)
            chunks = self.pipeline.ingest_bytes(
                source,
                job.document_id,
                source=job.filename,
                tenant_id=job.tenant_id or None,
            )
            self.repository.complete(job.id, len(chunks))
        except Exception as error:
            logger.exception(
                "ingestion_job_failed",
                extra={
                    "job_id": str(job.id),
                    "attempts": job.attempts,
                    "error": str(error),
                },
            )
            self.repository.fail(
                job.id,
                str(error),
                job.attempts,
                job.max_attempts,
            )
        return True

    def run_forever(self, poll_interval_seconds: float) -> None:
        while True:
            if not self.run_once():
                time.sleep(poll_interval_seconds)
