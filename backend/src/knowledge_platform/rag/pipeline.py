from pathlib import Path

from knowledge_platform.rag.chunking import TextChunker
from knowledge_platform.rag.embeddings import EmbeddingProvider
from knowledge_platform.rag.generation import ChatModelProvider, RAGAnswerer
from knowledge_platform.rag.loaders import PDFLoader
from knowledge_platform.rag.retrieval import Retriever
from knowledge_platform.rag.schemas import DocumentChunk, RAGAnswer
from knowledge_platform.rag.vector_store import VectorStore


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
        self.chunker = chunker if chunker is not None else TextChunker()
        self.retriever = Retriever(embeddings, store, collection, where, max_distance)
        self.answerer = RAGAnswerer(self.retriever, chat_model)

    def ingest(self, path: str | Path, document_id: str) -> list[DocumentChunk]:
        document = PDFLoader().load(path, document_id)
        chunks = self.chunker.chunk(document)
        vectors = self.embeddings.embed([chunk.content for chunk in chunks])
        self.store.add(self.collection, chunks, vectors)
        return chunks

    def answer(self, question: str, top_k: int = 5) -> RAGAnswer:
        return self.answerer.answer(question, top_k=top_k)
