import pytest

from knowledge_platform.rag.chunking import TextChunker
from knowledge_platform.rag.schemas import LoadedDocument, PageContent


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


def test_chunk_size_and_overlap() -> None:
    document = make_document([(1, "abcdefghij")])

    chunks = TextChunker(chunk_size=6, overlap=2).chunk(document)

    assert [chunk.content for chunk in chunks] == ["abcdef", "efghij"]
    assert [chunk.chunk_id for chunk in chunks] == ["chunk_001", "chunk_002"]
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
