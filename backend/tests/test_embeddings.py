from collections.abc import Sequence

import pytest

from knowledge_platform.exceptions import ServiceUnavailableError
from knowledge_platform.rag.embeddings import EmbeddingProvider, OllamaEmbeddings


class FakeClient:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[tuple[str, list[str]]] = []

    def embed(self, *, model: str, input: Sequence[str]) -> dict[str, list[list[float]]]:
        self.calls.append((model, list(input)))
        return {"embeddings": self.vectors}


def test_embed_returns_one_vector_per_text() -> None:
    client = FakeClient([[0.1, 0.2], [0.3, 0.4]])

    vectors = OllamaEmbeddings(client=client).embed(["a", "b"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert client.calls == [("nomic-embed-text", ["a", "b"])]


def test_custom_model_is_passed_to_client() -> None:
    client = FakeClient([[1.0]])

    OllamaEmbeddings(model="mxbai-embed-large", client=client).embed(["x"])

    assert client.calls[0][0] == "mxbai-embed-large"


def test_empty_input_does_not_call_client() -> None:
    client = FakeClient([])

    assert OllamaEmbeddings(client=client).embed([]) == []
    assert client.calls == []


def test_embed_one_returns_single_vector() -> None:
    client = FakeClient([[0.5, 0.6]])

    assert OllamaEmbeddings(client=client).embed_one("hello") == [0.5, 0.6]


class FlakyClient(FakeClient):
    def __init__(self, vectors: list[list[float]], failures: int) -> None:
        super().__init__(vectors)
        self.failures = failures
        self.attempts = 0

    def embed(
        self, *, model: str, input: Sequence[str]
    ) -> dict[str, list[list[float]]]:
        self.attempts += 1
        if self.attempts <= self.failures:
            raise ConnectionError("ollama down")
        return super().embed(model=model, input=input)


def test_provider_interface_is_abstract() -> None:
    with pytest.raises(TypeError):
        EmbeddingProvider()


def test_embed_retries_connection_errors() -> None:
    client = FlakyClient([[0.1]], failures=1)

    vectors = OllamaEmbeddings(
        client=client,
        retries=2,
        backoff_seconds=0,
    ).embed(["a"])

    assert vectors == [[0.1]]
    assert client.attempts == 2


def test_embed_maps_exhausted_retries_to_service_unavailable() -> None:
    client = FlakyClient([[0.1]], failures=2)

    with pytest.raises(ServiceUnavailableError):
        OllamaEmbeddings(
            client=client,
            retries=2,
            backoff_seconds=0,
        ).embed(["a"])

    assert client.attempts == 2
