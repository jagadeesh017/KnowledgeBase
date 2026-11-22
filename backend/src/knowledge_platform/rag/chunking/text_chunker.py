from collections.abc import Iterator

from knowledge_platform.rag.schemas.document import DocumentChunk, LoadedDocument


class TextChunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        if overlap < 0:
            raise ValueError("overlap cannot be negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: LoadedDocument) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        counter = 0

        for page in document.pages:
            text = page.text.strip()
            if not text:
                continue

            for piece in self._split(text):
                counter += 1
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"chunk_{counter:03d}",
                        document_id=document.document_id,
                        content=piece,
                        page_number=page.page_number,
                        metadata={
                            "source": document.source,
                            **document.metadata,
                            **page.metadata,
                        },
                    )
                )

        return chunks

    def _split(self, text: str) -> Iterator[str]:
        start = 0
        step = self.chunk_size - self.overlap
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            yield text[start:end]
            if end == len(text):
                break
            start += step
