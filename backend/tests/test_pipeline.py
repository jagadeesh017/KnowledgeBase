from pathlib import Path
from uuid import uuid4

import chromadb
import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from knowledge_platform.rag.chunking import TextChunker
from knowledge_platform.rag.embeddings import EmbeddingProvider
from knowledge_platform.rag.generation import ChatMessage, ChatModelProvider
from knowledge_platform.rag.pipeline import RAGPipeline
from knowledge_platform.rag.vector_store import ChromaVectorStore


class FakeEmbeddings(EmbeddingProvider):
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class FakeChatModel(ChatModelProvider):
    def chat(self, messages: list[ChatMessage]) -> str:
        return "stub answer [1]"


@pytest.fixture
def pdf_path(tmp_path: Path) -> Path:
    path = tmp_path / "guide.pdf"
    pdf = canvas.Canvas(str(path), pagesize=letter)
    pdf.drawString(72, 750, "RAG stores document chunks as vectors.")
    pdf.showPage()
    pdf.save()
    return path


def test_pipeline_ingests_then_answers(pdf_path: Path) -> None:
    store = ChromaVectorStore(client=chromadb.EphemeralClient())
    pipeline = RAGPipeline(
        FakeEmbeddings(),
        FakeChatModel(),
        store,
        f"kb_{uuid4().hex[:12]}",
        chunker=TextChunker(chunk_size=200, overlap=0),
    )

    chunks = pipeline.ingest(pdf_path, "doc_001")
    answer = pipeline.answer("What does RAG store?")

    assert len(chunks) == 1
    assert answer.answer == "stub answer [1]"
    assert answer.sources[0].page_number == 1
    assert answer.sources[0].source == "guide.pdf"
