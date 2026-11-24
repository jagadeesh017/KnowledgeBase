from functools import cache

from config import Settings
from db.postgres import (
    PsycopgConnectionProvider,
    apply_migrations,
)
from jobs.repository import IngestionJobRepository
from rag.chunking import TextChunker
from rag.embeddings import OllamaEmbeddings
from rag.generation import OllamaChatModel
from rag.pipeline import RAGPipeline
from rag.vector_store import (
    ChromaVectorStore,
    PgVectorStore,
    VectorStore,
)


@cache
def get_connection_provider(
    dsn: str,
    min_size: int,
    max_size: int,
) -> PsycopgConnectionProvider:
    return PsycopgConnectionProvider(dsn, min_size=min_size, max_size=max_size)


def provider_from_settings(settings: Settings) -> PsycopgConnectionProvider:
    return get_connection_provider(
        settings.database_url,
        settings.db_pool_min_size,
        settings.db_pool_max_size,
    )


def build_pipeline(
    settings: Settings,
    provider: PsycopgConnectionProvider | None = None,
) -> RAGPipeline:
    embeddings = OllamaEmbeddings(
        model=settings.embedding_model,
        host=settings.ollama_host,
        timeout_seconds=settings.ollama_timeout_seconds,
        retries=settings.ollama_retries,
        backoff_seconds=settings.ollama_retry_backoff_seconds,
    )
    if settings.vector_store_backend == "pgvector":
        active_provider = provider or provider_from_settings(settings)
        apply_migrations(active_provider)
        store: VectorStore = PgVectorStore(active_provider)
    else:
        store = ChromaVectorStore(path=settings.chroma_path)

    return RAGPipeline(
        embeddings=embeddings,
        chat_model=OllamaChatModel(
            model=settings.chat_model,
            host=settings.ollama_host,
            timeout_seconds=settings.ollama_timeout_seconds,
            retries=settings.ollama_retries,
            backoff_seconds=settings.ollama_retry_backoff_seconds,
        ),
        store=store,
        collection=settings.default_collection,
        chunker=TextChunker(
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
            embeddings=embeddings,
            semantic_break_quantile=settings.semantic_break_quantile,
            embedding_batch_size=settings.embedding_batch_size,
        ),
        max_distance=settings.max_distance,
    )


def build_job_repository(
    settings: Settings,
    provider: PsycopgConnectionProvider | None = None,
) -> IngestionJobRepository:
    active_provider = provider or provider_from_settings(settings)
    apply_migrations(active_provider)
    return IngestionJobRepository(
        active_provider,
        retry_backoff_seconds=settings.job_retry_backoff_seconds,
    )
