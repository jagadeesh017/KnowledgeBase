from uuid import uuid4

import chromadb

from knowledge_platform.rag.embeddings import EmbeddingProvider
from knowledge_platform.rag.retrieval import Retriever
from knowledge_platform.rag.schemas import DocumentChunk, SearchResult
from knowledge_platform.rag.vector_store import ChromaVectorStore, VectorStore


class FakeEmbeddings(EmbeddingProvider):
    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.queries: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.queries.extend(texts)
        return [list(self.vector) for _ in texts]


class FakeStore(VectorStore):
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, list[float], int, dict[str, object] | None]] = []

    def add(
        self,
        collection: str,
        chunks: list[DocumentChunk],
        vectors: list[list[float]],
    ) -> None:
        raise NotImplementedError

    def query(
        self,
        collection: str,
        vector: list[float],
        top_k: int = 5,
        where: dict[str, object] | None = None,
    ) -> list[SearchResult]:
        self.calls.append((collection, vector, top_k, where))
        return self.results

    def delete(self, collection: str, where: dict[str, object]) -> None:
        raise NotImplementedError


def make_result(distance: float, document_id: str = "doc_001") -> SearchResult:
    return SearchResult(
        chunk_id="chunk_001",
        document_id=document_id,
        content="alpha",
        page_number=1,
        distance=distance,
        metadata={"source": "guide.pdf"},
    )


def make_chunk(chunk_id: str, document_id: str, content: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        content=content,
        page_number=1,
        metadata={"source": f"{document_id}.pdf"},
    )


def test_retrieve_embeds_query_and_queries_store() -> None:
    embeddings = FakeEmbeddings([1.0, 0.0])
    store = FakeStore([make_result(0.1)])

    results = Retriever(embeddings, store, "kb_001").retrieve("question", top_k=3)

    assert embeddings.queries == ["question"]
    assert store.calls == [("kb_001", [1.0, 0.0], 3, None)]
    assert results == store.results


def test_where_filter_is_passed_to_store() -> None:
    store = FakeStore([])
    where = {"document_id": "doc_001"}

    Retriever(FakeEmbeddings([1.0]), store, "kb_001", where=where).retrieve("q")

    assert store.calls[0][3] == where


def test_max_distance_filters_results() -> None:
    store = FakeStore([make_result(0.2), make_result(0.9)])

    results = Retriever(FakeEmbeddings([1.0]), store, "kb_001", max_distance=0.5).retrieve(
        "q"
    )

    assert [result.distance for result in results] == [0.2]


def test_integration_retriever_isolates_by_collection() -> None:
    store = ChromaVectorStore(client=chromadb.EphemeralClient())
    first = f"kb_{uuid4().hex[:12]}_a"
    second = f"kb_{uuid4().hex[:12]}_b"
    store.add(first, [make_chunk("chunk_a", "doc_a", "alpha")], [[1.0, 0.0]])
    store.add(second, [make_chunk("chunk_b", "doc_b", "beta")], [[1.0, 0.0]])

    results = Retriever(FakeEmbeddings([1.0, 0.0]), store, first).retrieve("q")

    assert [result.document_id for result in results] == ["doc_a"]


def test_integration_where_filter_limits_results() -> None:
    store = ChromaVectorStore(client=chromadb.EphemeralClient())
    collection = f"kb_{uuid4().hex[:12]}"
    store.add(
        collection,
        [
            make_chunk("chunk_a", "doc_a", "alpha"),
            make_chunk("chunk_b", "doc_b", "beta"),
        ],
        [[1.0, 0.0], [0.9, 0.1]],
    )

    retriever = Retriever(
        FakeEmbeddings([1.0, 0.0]), store, collection, where={"document_id": "doc_b"}
    )

    assert [result.document_id for result in retriever.retrieve("q")] == ["doc_b"]
