import pytest

from rag.embeddings import EmbeddingProvider
from rag.generation import (
    ChatMessage,
    ChatModelProvider,
    OllamaChatModel,
    RAGAnswerer,
)
from rag.generation.prompts import SYSTEM_PROMPT, build_prompt
from rag.retrieval import Retriever
from rag.schemas import SearchResult
from rag.vector_store import VectorStore


class FakeChatModel(ChatModelProvider):
    def __init__(self, reply: str = "answer [1]") -> None:
        self.reply = reply
        self.calls: list[list[ChatMessage]] = []

    def chat(self, messages: list[ChatMessage]) -> str:
        self.calls.append(messages)
        return self.reply


class FakeEmbeddings(EmbeddingProvider):
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]


class FakeStore(VectorStore):
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results

    def add(self, collection, chunks, vectors) -> None:
        raise NotImplementedError

    def query(self, collection, vector, top_k=5, where=None):
        return self.results

    def delete(self, collection, where) -> None:
        raise NotImplementedError


def make_result() -> SearchResult:
    return SearchResult(
        chunk_id="chunk_001",
        document_id="doc_001",
        content="alpha",
        page_number=2,
        distance=0.1,
        metadata={"source": "guide.pdf"},
    )


def test_ollama_chat_model_sends_messages() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.calls = []

        def chat(self, *, model, messages):
            self.calls.append((model, messages))
            return {"message": {"content": "hi"}}

    client = FakeClient()

    reply = OllamaChatModel(model="qwen2.5:3b", client=client).chat(
        [ChatMessage(role="user", content="hello")]
    )

    assert reply == "hi"
    assert client.calls == [("qwen2.5:3b", [{"role": "user", "content": "hello"}])]


def test_build_prompt_includes_context_and_question() -> None:
    prompt = build_prompt("What is RAG?", "[1] guide.pdf (page 2)\nalpha")

    assert "Context:" in prompt
    assert "[1] guide.pdf (page 2)" in prompt
    assert "What is RAG?" in prompt


def test_build_prompt_without_context() -> None:
    prompt = build_prompt("What is RAG?", "")

    assert "No context blocks were retrieved" in prompt


def test_answerer_returns_answer_and_sources() -> None:
    model = FakeChatModel("alpha comes from [1]")
    answerer = RAGAnswerer(
        Retriever(FakeEmbeddings(), FakeStore([make_result()]), "kb_001"), model
    )

    answer = answerer.answer("What is alpha?")

    assert answer.answer == "alpha comes from [1]"
    assert len(answer.sources) == 1
    assert answer.sources[0].chunk_id == "chunk_001"
    assert answer.sources[0].page_number == 2
    assert answer.sources[0].source == "guide.pdf"


def test_answerer_sends_system_prompt_and_context() -> None:
    model = FakeChatModel()
    answerer = RAGAnswerer(
        Retriever(FakeEmbeddings(), FakeStore([make_result()]), "kb_001"), model
    )

    answerer.answer("question")

    messages = model.calls[0]
    assert messages[0].role == "system"
    assert messages[0].content == SYSTEM_PROMPT
    assert "alpha" in messages[1].content


def test_answerer_with_no_results() -> None:
    model = FakeChatModel("I cannot answer")
    answerer = RAGAnswerer(
        Retriever(FakeEmbeddings(), FakeStore([]), "kb_001"), model
    )

    answer = answerer.answer("question")

    assert answer.sources == []
    assert "No context blocks" in model.calls[0][1].content


def test_chat_model_interface_is_abstract() -> None:
    with pytest.raises(TypeError):
        ChatModelProvider()
