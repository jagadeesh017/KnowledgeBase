from collections.abc import Sequence
from typing import Any, Protocol, cast

import ollama

from rag.embeddings.base import EmbeddingProvider
from rag.ollama_retry import call_with_retries


class EmbedResponse(Protocol):
    def __getitem__(self, key: str) -> list[list[float]]: ...


class EmbedClient(Protocol):
    def embed(self, *, model: str, input: Sequence[str]) -> EmbedResponse: ...


class OllamaEmbeddings(EmbeddingProvider):
    DEFAULT_MODEL = "nomic-embed-text"
    DEFAULT_HOST = "http://localhost:11434"
    DEFAULT_TIMEOUT_SECONDS = 60.0
    DEFAULT_RETRIES = 2
    DEFAULT_BACKOFF_SECONDS = 0.5

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        client: EmbedClient | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        retries: int = DEFAULT_RETRIES,
        backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    ) -> None:
        self.model = model
        self.retries = max(1, retries)
        self.backoff_seconds = backoff_seconds
        self.client = (
            client
            if client is not None
            else ollama.Client(host=host, timeout=timeout_seconds)
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = call_with_retries(
            lambda: self.client.embed(model=self.model, input=texts),
            attempts=self.retries,
            backoff_seconds=self.backoff_seconds,
        )
        return [list(vector) for vector in response["embeddings"]]

    def check_ready(self) -> None:
        list_client = cast(Any, self.client)
        call_with_retries(
            lambda: list_client.list(),
            attempts=self.retries,
            backoff_seconds=self.backoff_seconds,
        )
