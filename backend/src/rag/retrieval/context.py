from rag.schemas import SearchResult, Source


def build_context(results: list[SearchResult]) -> str:
    blocks = []
    for index, result in enumerate(results, start=1):
        source = result.metadata.get("source", "unknown")
        blocks.append(
            f"[{index}] {source} (page {result.page_number})\n{result.content}"
        )
    return "\n\n".join(blocks)


def build_sources(results: list[SearchResult]) -> list[Source]:
    return [
        Source(
            index=index,
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            source=result.metadata.get("source", "unknown"),
            page_number=result.page_number,
            distance=result.distance,
        )
        for index, result in enumerate(results, start=1)
    ]
