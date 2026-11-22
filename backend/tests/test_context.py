from knowledge_platform.rag.retrieval import build_context, build_sources
from knowledge_platform.rag.schemas import SearchResult


def make_result(
    chunk_id: str,
    document_id: str,
    content: str,
    page_number: int,
    metadata: dict[str, str] | None = None,
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        document_id=document_id,
        content=content,
        page_number=page_number,
        distance=0.25,
        metadata=metadata if metadata is not None else {"source": "guide.pdf"},
    )


def test_build_context_numbers_and_labels_sources() -> None:
    results = [
        make_result("chunk_001", "doc_001", "first chunk", 1),
        make_result("chunk_002", "doc_002", "second chunk", 3),
    ]

    assert build_context(results) == (
        "[1] guide.pdf (page 1)\nfirst chunk\n\n"
        "[2] guide.pdf (page 3)\nsecond chunk"
    )


def test_build_sources_matches_context_indices() -> None:
    results = [
        make_result("chunk_001", "doc_001", "first chunk", 1),
        make_result("chunk_002", "doc_002", "second chunk", 3),
    ]

    sources = build_sources(results)

    assert [source.index for source in sources] == [1, 2]
    assert sources[0].chunk_id == "chunk_001"
    assert sources[0].document_id == "doc_001"
    assert sources[0].source == "guide.pdf"
    assert sources[0].page_number == 1
    assert sources[0].distance == 0.25


def test_missing_source_defaults_to_unknown() -> None:
    results = [make_result("chunk_001", "doc_001", "text", 1, metadata={})]

    assert build_context(results) == "[1] unknown (page 1)\ntext"
    assert build_sources(results)[0].source == "unknown"


def test_empty_results_produce_empty_output() -> None:
    assert build_context([]) == ""
    assert build_sources([]) == []
