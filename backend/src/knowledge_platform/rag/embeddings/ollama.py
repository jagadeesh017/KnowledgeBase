from collections.abc import Sequence
from typing import Protocol

import ollama

from knowledge_platform.rag.embeddings.base import EmbeddingProvider


class EmbedResponse(Protocol):
    def __getitem__(self, key: str) -> list[list[float]]: ...


class EmbedClient(Protocol):
    def embed(self, *, model: str, input: Sequence[str]) -> EmbedResponse: ...


class OllamaEmbeddings(EmbeddingProvider):
    DEFAULT_MODEL = "nomic-embed-text"
    DEFAULT_HOST = "http://localhost:11434"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        client: EmbedClient | None = None,
    ) -> None:
        self.model = model
        self.client = client if client is not None else ollama.Client(host=host)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embed(model=self.model, input=texts)
        return [list(vector) for vector in response["embeddings"]]
