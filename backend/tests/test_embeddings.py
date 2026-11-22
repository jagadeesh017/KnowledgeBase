from collections.abc import Sequence

import pytest

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


def test_provider_interface_is_abstract() -> None:
    with pytest.raises(TypeError):
        EmbeddingProvider()
