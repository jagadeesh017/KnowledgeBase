from pathlib import Path
from uuid import uuid4

from exceptions import EmptyDocumentError
from rag.chunking import TextChunker
from rag.embeddings import EmbeddingProvider
from rag.generation import ChatModelProvider, RAGAnswerer
from rag.loaders import PDFLoader
from rag.retrieval import Retriever
from rag.schemas import DocumentChunk, RAGAnswer
from rag.vector_store import VectorStore


class RAGPipeline:
    def __init__(
        self,
        embeddings: EmbeddingProvider,
        chat_model: ChatModelProvider,
        store: VectorStore,
        collection: str,
        chunker: TextChunker | None = None,
        where: dict[str, object] | None = None,
        max_distance: float | None = None,
    ) -> None:
        self.embeddings = embeddings
        self.store = store
        self.collection = collection
        self.chunker = (
            chunker if chunker is not None else TextChunker(embeddings=embeddings)
        )
        self.retriever = Retriever(embeddings, store, collection, where, max_distance)
        self.answerer = RAGAnswerer(self.retriever, chat_model)

    def ingest(
        self,
        path: str | Path,
        document_id: str,
        *,
        tenant_id: str | None = None,
    ) -> list[DocumentChunk]:
        pdf_path = Path(path)
        return self.ingest_bytes(
            pdf_path.read_bytes(),
            document_id,
            source=pdf_path.name,
            tenant_id=tenant_id,
        )

    def ingest_bytes(
        self,
        data: bytes,
        document_id: str,
        *,
        source: str = "document.pdf",
        tenant_id: str | None = None,
    ) -> list[DocumentChunk]:
        document = PDFLoader().load_bytes(data, document_id, source=source)
        chunks = self.chunker.chunk(document)
        if not chunks:
            raise EmptyDocumentError(
                f"Document {document_id} has no extractable text"
            )

        version = uuid4().hex
        prefix = f"{tenant_id}:" if tenant_id else ""
        for index, chunk in enumerate(chunks):
            chunk.chunk_id = f"{prefix}{document_id}:{version}:{index:04d}"
            chunk.metadata["ingest_version"] = version
            if tenant_id:
                chunk.metadata["tenant_id"] = tenant_id

        vectors = self.embeddings.embed([chunk.content for chunk in chunks])
        self.store.add(self.collection, chunks, vectors)
        self.store.delete(
            self.collection,
            self._replacement_where(document_id, version, tenant_id),
        )
        return chunks

    def answer(
        self,
        question: str,
        top_k: int = 5,
        where: dict[str, object] | None = None,
        *,
        tenant_id: str | None = None,
    ) -> RAGAnswer:
        return self.answerer.answer(
            question,
            top_k=top_k,
            where=self._scoped_where(where, tenant_id),
        )

    def delete_document(
        self,
        document_id: str,
        *,
        tenant_id: str | None = None,
    ) -> None:
        delete_where = self._scoped_where(
            {"document_id": document_id},
            tenant_id,
        )
        assert delete_where is not None
        self.store.delete(self.collection, delete_where)

    def check_ready(self) -> None:
        self.embeddings.check_ready()
        self.store.check_ready()

    @staticmethod
    def _replacement_where(
        document_id: str,
        version: str,
        tenant_id: str | None,
    ) -> dict[str, object]:
        conditions: list[dict[str, object]] = [
            {"document_id": document_id},
            {"ingest_version": {"$ne": version}},
        ]
        if tenant_id:
            conditions.append({"tenant_id": tenant_id})
        return {"$and": conditions}

    @staticmethod
    def _scoped_where(
        where: dict[str, object] | None,
        tenant_id: str | None,
    ) -> dict[str, object] | None:
        if tenant_id is None:
            return where
        tenant_filter: dict[str, object] = {"tenant_id": tenant_id}
        if where is None:
            return tenant_filter
        return {"$and": [where, tenant_filter]}
