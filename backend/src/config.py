from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "postgresql://postgres:postgres@localhost:5432/knowledge"
    vector_store_backend: Literal["chroma", "pgvector"] = "pgvector"
    db_pool_min_size: int = Field(default=1, ge=1)
    db_pool_max_size: int = Field(default=10, ge=1)
    job_poll_interval_seconds: float = Field(default=1.0, gt=0)
    job_max_attempts: int = Field(default=3, ge=1)
    job_retry_backoff_seconds: float = Field(default=5.0, gt=0)
    ollama_host: str = "http://localhost:11434"
    ollama_timeout_seconds: float = Field(default=60.0, gt=0)
    ollama_retries: int = Field(default=2, ge=1)
    ollama_retry_backoff_seconds: float = Field(default=0.5, ge=0)
    max_upload_bytes: int = Field(default=20 * 1024 * 1024, gt=0)
    embedding_model: str = "nomic-embed-text"
    chat_model: str = "qwen2.5:3b"
    chroma_path: Path = Path("./chroma")
    default_collection: str = "knowledge"
    chunk_size: int = Field(default=500, gt=0)
    chunk_overlap: int = Field(default=50, ge=0)
    top_k: int = Field(default=5, gt=0)
    max_distance: float | None = Field(default=None, ge=0)
    semantic_break_quantile: float = Field(default=0.75, gt=0, lt=1)
    embedding_batch_size: int = Field(default=64, gt=0)
    auth_enabled: bool = True
    api_keys: dict[str, str] = Field(default_factory=dict)
    log_level: str = "INFO"

    @model_validator(mode="after")
    def validate_chunking(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("KP_CHUNK_OVERLAP must be smaller than KP_CHUNK_SIZE")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
