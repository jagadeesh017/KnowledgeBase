import pytest

from rag.chunking import TextChunker
from rag.embeddings import EmbeddingProvider
from rag.schemas import LoadedDocument, PageContent


def make_document(
    pages: list[tuple[int, str]],
    document_id: str = "doc_001",
    source: str = "guide.pdf",
    metadata: dict[str, str] | None = None,
) -> LoadedDocument:
    return LoadedDocument(
        document_id=document_id,
        source=source,
        pages=[
            PageContent(page_number=number, text=text) for number, text in pages
        ],
        metadata=metadata or {},
    )


class FakeEmbeddings(EmbeddingProvider):
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.batches: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return [self.vectors[text] for text in texts]


def test_chunk_size_and_overlap() -> None:
    document = make_document([(1, "abcdefghij")])

    chunks = TextChunker(chunk_size=6, overlap=2).chunk(document)

    assert [chunk.content for chunk in chunks] == ["abcdef", "efghij"]
    assert [chunk.chunk_id for chunk in chunks] == [
        "doc_001:chunk_001",
        "doc_001:chunk_002",
    ]
    assert all(chunk.page_number == 1 for chunk in chunks)
    assert all(chunk.document_id == "doc_001" for chunk in chunks)
    assert all(chunk.metadata["source"] == "guide.pdf" for chunk in chunks)


def test_chunks_never_cross_page_boundaries() -> None:
    document = make_document([(1, "AAAAAAAAAA"), (2, "BBBBBBBBBB")])

    chunks = TextChunker(chunk_size=6, overlap=2).chunk(document)

    assert [chunk.page_number for chunk in chunks] == [1, 1, 2, 2]
    assert [chunk.content for chunk in chunks] == [
        "AAAAAA",
        "AAAAAA",
        "BBBBBB",
        "BBBBBB",
    ]


def test_empty_pages_produce_no_chunks() -> None:
    document = make_document([(1, ""), (2, "   "), (3, "abc")])

    chunks = TextChunker(chunk_size=5, overlap=1).chunk(document)

    assert [chunk.page_number for chunk in chunks] == [3]


def test_metadata_is_propagated() -> None:
    document = make_document([(1, "abc")], metadata={"Title": "Guide"})

    chunks = TextChunker(chunk_size=3, overlap=0).chunk(document)

    assert chunks[0].metadata == {"source": "guide.pdf", "Title": "Guide"}


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (-1, 0), (5, -1), (5, 5), (5, 6)],
)
def test_invalid_configuration(chunk_size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        TextChunker(chunk_size=chunk_size, overlap=overlap)


def test_structural_chunking_keeps_sentences_intact() -> None:
    document = make_document(
        [(1, "Alpha beta. Gamma delta. Epsilon zeta.")]
    )

    chunks = TextChunker(chunk_size=20, overlap=0).chunk(document)

    assert [chunk.content for chunk in chunks] == [
        "Alpha beta.",
        "Gamma delta.",
        "Epsilon zeta.",
    ]


def test_semantic_break_splits_on_topic_shift() -> None:
    document = make_document(
        [(1, "Alpha beta. Gamma delta. Epsilon zeta.")]
    )
    embeddings = FakeEmbeddings(
        {
            "Alpha beta.": [1.0, 0.0],
            "Gamma delta.": [0.9, 0.1],
            "Epsilon zeta.": [0.0, 1.0],
        }
    )

    chunks = TextChunker(chunk_size=100, overlap=0, embeddings=embeddings).chunk(document)

    assert [chunk.content for chunk in chunks] == [
        "Alpha beta. Gamma delta.",
        "Epsilon zeta.",
    ]


def test_semantic_embeddings_are_batched() -> None:
    document = make_document(
        [(1, "Alpha beta. Gamma delta. Epsilon zeta.")]
    )
    embeddings = FakeEmbeddings(
        {
            "Alpha beta.": [1.0, 0.0],
            "Gamma delta.": [0.9, 0.1],
            "Epsilon zeta.": [0.0, 1.0],
        }
    )

    TextChunker(
        chunk_size=100,
        overlap=0,
        embeddings=embeddings,
        embedding_batch_size=2,
    ).chunk(document)

    assert embeddings.batches == [
        ["Alpha beta.", "Gamma delta."],
        ["Epsilon zeta."],
    ]


def test_paragraph_boundaries_are_preserved() -> None:
    document = make_document(
        [(1, "First paragraph sentence.\n\nSecond paragraph sentence.")]
    )

    chunks = TextChunker(chunk_size=200, overlap=0).chunk(document)

    assert [chunk.content for chunk in chunks] == [
        "First paragraph sentence.",
        "Second paragraph sentence.",
    ]


def test_invalid_semantic_configuration() -> None:
    with pytest.raises(ValueError):
        TextChunker(semantic_break_quantile=0)
    with pytest.raises(ValueError):
        TextChunker(semantic_break_quantile=1)
    with pytest.raises(ValueError):
        TextChunker(embedding_batch_size=0)
