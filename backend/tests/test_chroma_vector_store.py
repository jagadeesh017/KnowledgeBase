from uuid import uuid4

import chromadb
import pytest

from knowledge_platform.rag.schemas import DocumentChunk
from knowledge_platform.rag.vector_store import ChromaVectorStore, VectorStore


@pytest.fixture
def store() -> ChromaVectorStore:
    return ChromaVectorStore(client=chromadb.EphemeralClient())


@pytest.fixture
def collection_name() -> str:
    return f"kb_{uuid4().hex[:12]}"


def make_chunk(
    chunk_id: str,
    document_id: str,
    content: str,
    page_number: int = 1,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        content=content,
        page_number=page_number,
        metadata={"source": "guide.pdf"},
    )


def test_add_and_query_round_trip(
    store: ChromaVectorStore, collection_name: str
) -> None:
    store.add(
        collection_name,
        [make_chunk("chunk_001", "doc_001", "hello world", 2)],
        [[1.0, 0.0]],
    )

    results = store.query(collection_name, [1.0, 0.0])

    assert len(results) == 1
    assert results[0].chunk_id == "chunk_001"
    assert results[0].document_id == "doc_001"
    assert results[0].content == "hello world"
    assert results[0].page_number == 2
    assert results[0].distance == pytest.approx(0.0)
    assert results[0].metadata == {"source": "guide.pdf"}


def test_query_missing_collection_returns_empty(
    store: ChromaVectorStore, collection_name: str
) -> None:
    assert store.query(collection_name, [1.0, 0.0]) == []


def test_where_filter_limits_results(
    store: ChromaVectorStore, collection_name: str
) -> None:
    chunks = [
        make_chunk("chunk_001", "doc_001", "alpha"),
        make_chunk("chunk_002", "doc_002", "beta"),
    ]
    store.add(collection_name, chunks, [[1.0, 0.0], [0.0, 1.0]])

    results = store.query(collection_name, [1.0, 0.0], where={"document_id": "doc_002"})

    assert [result.document_id for result in results] == ["doc_002"]


def test_top_k_limits_results(store: ChromaVectorStore, collection_name: str) -> None:
    chunks = [make_chunk(f"chunk_{i:03d}", "doc_001", f"text {i}") for i in range(1, 4)]
    store.add(collection_name, chunks, [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]])

    assert len(store.query(collection_name, [1.0, 0.0], top_k=2)) == 2


def test_delete_by_where(store: ChromaVectorStore, collection_name: str) -> None:
    chunks = [
        make_chunk("chunk_001", "doc_001", "alpha"),
        make_chunk("chunk_002", "doc_002", "beta"),
    ]
    store.add(collection_name, chunks, [[1.0, 0.0], [0.0, 1.0]])

    store.delete(collection_name, {"document_id": "doc_001"})

    assert (
        store.query(collection_name, [1.0, 0.0], where={"document_id": "doc_001"}) == []
    )
    assert len(store.query(collection_name, [1.0, 0.0])) == 1


def test_collections_are_isolated(store: ChromaVectorStore) -> None:
    first = f"kb_{uuid4().hex[:12]}_a"
    second = f"kb_{uuid4().hex[:12]}_b"
    store.add(first, [make_chunk("chunk_001", "doc_001", "secret a")], [[1.0, 0.0]])
    store.add(second, [make_chunk("chunk_001", "doc_002", "secret b")], [[1.0, 0.0]])

    results = store.query(first, [1.0, 0.0])

    assert [result.document_id for result in results] == ["doc_001"]


def test_readding_chunk_id_replaces_content(
    store: ChromaVectorStore, collection_name: str
) -> None:
    store.add(collection_name, [make_chunk("chunk_001", "doc_001", "old text")], [[1.0, 0.0]])
    store.add(collection_name, [make_chunk("chunk_001", "doc_001", "new text")], [[1.0, 0.0]])

    results = store.query(collection_name, [1.0, 0.0])

    assert [result.content for result in results] == ["new text"]


def test_mismatched_lengths_raise(
    store: ChromaVectorStore, collection_name: str
) -> None:
    with pytest.raises(ValueError):
        store.add(collection_name, [make_chunk("chunk_001", "doc_001", "x")], [])


def test_empty_add_is_noop(store: ChromaVectorStore, collection_name: str) -> None:
    store.add(collection_name, [], [])
    assert store.query(collection_name, [1.0, 0.0]) == []


def test_vector_store_interface_is_abstract() -> None:
    with pytest.raises(TypeError):
        VectorStore()
