import os

import pytest

from knowledge_platform.rag.embeddings import OllamaEmbeddings

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_TESTS") != "1",
    reason="set RUN_LIVE_TESTS=1 to run against a live Ollama server",
)


def test_live_embedding_from_ollama() -> None:
    vectors = OllamaEmbeddings().embed(["hello world", "rag guide"])

    assert len(vectors) == 2
    assert all(len(vector) > 0 for vector in vectors)
    assert all(isinstance(value, float) for vector in vectors for value in vector)
