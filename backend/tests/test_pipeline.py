from pathlib import Path
from uuid import uuid4

import chromadb
import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from knowledge_platform.exceptions import EmptyDocumentError, ServiceUnavailableError
from knowledge_platform.rag.chunking import TextChunker
from knowledge_platform.rag.embeddings import EmbeddingProvider
from knowledge_platform.rag.generation import ChatMessage, ChatModelProvider
from knowledge_platform.rag.pipeline import RAGPipeline
from knowledge_platform.rag.schemas import DocumentChunk, SearchResult
from knowledge_platform.rag.vector_store import ChromaVectorStore, VectorStore


class FakeEmbeddings(EmbeddingProvider):
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class FakeChatModel(ChatModelProvider):
    def chat(self, messages: list[ChatMessage]) -> str:
        return "stub answer [1]"


class FailingDeleteStore(VectorStore):
    def __init__(self, inner: VectorStore, fail_on_delete: int = 2) -> None:
        self.inner = inner
        self.fail_on_delete = fail_on_delete
        self.deletes = 0

    def add(
        self,
        collection: str,
        chunks: list[DocumentChunk],
        vectors: list[list[float]],
    ) -> None:
        self.inner.add(collection, chunks, vectors)

    def query(
        self,
        collection: str,
        vector: list[float],
        top_k: int = 5,
        where: dict[str, object] | None = None,
    ) -> list[SearchResult]:
        return self.inner.query(collection, vector, top_k=top_k, where=where)

    def delete(self, collection: str, where: dict[str, object]) -> None:
        self.deletes += 1
        if self.deletes == self.fail_on_delete:
            raise ServiceUnavailableError("cleanup failed")
        self.inner.delete(collection, where)


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


def test_reingesting_document_replaces_old_chunks(
    pdf_path: Path, tmp_path: Path
) -> None:
    store = ChromaVectorStore(client=chromadb.EphemeralClient())
    pipeline = RAGPipeline(
        FakeEmbeddings(),
        FakeChatModel(),
        store,
        f"kb_{uuid4().hex[:12]}",
        chunker=TextChunker(chunk_size=200, overlap=0),
    )

    first = pipeline.ingest(pdf_path, "doc_001")
    replacement = tmp_path / "replacement.pdf"
    pdf = canvas.Canvas(str(replacement), pagesize=letter)
    pdf.drawString(72, 750, "Replacement text.")
    pdf.showPage()
    pdf.save()
    second = pipeline.ingest(replacement, "doc_001")

    assert first[0].content != second[0].content
    results = store.query(
        pipeline.collection, [1.0, 0.0], where={"document_id": "doc_001"}
    )
    assert len(results) == 1
    assert results[0].content == "Replacement text."


def test_failed_cleanup_keeps_old_and_new_chunks(
    pdf_path: Path, tmp_path: Path
) -> None:
    store = ChromaVectorStore(client=chromadb.EphemeralClient())
    pipeline = RAGPipeline(
        FakeEmbeddings(),
        FakeChatModel(),
        FailingDeleteStore(store),
        f"kb_{uuid4().hex[:12]}",
        chunker=TextChunker(chunk_size=200, overlap=0),
    )
    pipeline.ingest(pdf_path, "doc_001")

    replacement = tmp_path / "replacement.pdf"
    pdf = canvas.Canvas(str(replacement), pagesize=letter)
    pdf.drawString(72, 750, "Replacement text.")
    pdf.showPage()
    pdf.save()

    with pytest.raises(ServiceUnavailableError):
        pipeline.ingest(replacement, "doc_001")

    results = store.query(
        pipeline.collection, [1.0, 0.0], where={"document_id": "doc_001"}
    )
    assert sorted(result.content for result in results) == [
        "RAG stores document chunks as vectors.",
        "Replacement text.",
    ]


def test_blank_pdf_raises_empty_document_error(tmp_path: Path) -> None:
    blank = tmp_path / "blank.pdf"
    pdf = canvas.Canvas(str(blank), pagesize=letter)
    pdf.showPage()
    pdf.save()
    pipeline = RAGPipeline(
        FakeEmbeddings(),
        FakeChatModel(),
        ChromaVectorStore(client=chromadb.EphemeralClient()),
        f"kb_{uuid4().hex[:12]}",
    )

    with pytest.raises(EmptyDocumentError):
        pipeline.ingest(blank, "doc_blank")


def test_tenant_isolation_prevents_overwrite_and_deletion(tmp_path: Path) -> None:
    store = ChromaVectorStore(client=chromadb.EphemeralClient())
    pipeline = RAGPipeline(
        FakeEmbeddings(),
        FakeChatModel(),
        store,
        f"kb_{uuid4().hex[:12]}",
        chunker=TextChunker(chunk_size=200, overlap=0),
    )

    first = tmp_path / "tenant_a.pdf"
    pdf = canvas.Canvas(str(first), pagesize=letter)
    pdf.drawString(72, 750, "Tenant A content.")
    pdf.showPage()
    pdf.save()

    second = tmp_path / "tenant_b.pdf"
    pdf = canvas.Canvas(str(second), pagesize=letter)
    pdf.drawString(72, 750, "Tenant B content.")
    pdf.showPage()
    pdf.save()

    pipeline.ingest(first, "doc_shared", tenant_id="tenant_a")
    pipeline.ingest(second, "doc_shared", tenant_id="tenant_b")

    tenant_a_results = store.query(
        pipeline.collection, [1.0, 0.0], where={"tenant_id": "tenant_a"}
    )
    tenant_b_results = store.query(
        pipeline.collection, [1.0, 0.0], where={"tenant_id": "tenant_b"}
    )
    assert [result.content for result in tenant_a_results] == ["Tenant A content."]
    assert [result.content for result in tenant_b_results] == ["Tenant B content."]
    assert all(
        result.chunk_id.startswith("tenant_a:") for result in tenant_a_results
    )

    answer = pipeline.answer(
        "question",
        where={"document_id": "doc_shared"},
        tenant_id="tenant_a",
    )
    assert len(answer.sources) == 1

    pipeline.delete_document("doc_shared", tenant_id="tenant_a")

    assert (
        store.query(
            pipeline.collection, [1.0, 0.0], where={"tenant_id": "tenant_a"}
        )
        == []
    )
    assert (
        len(
            store.query(
                pipeline.collection, [1.0, 0.0], where={"tenant_id": "tenant_b"}
            )
        )
        == 1
    )
