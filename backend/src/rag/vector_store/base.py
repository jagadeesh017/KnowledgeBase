from abc import ABC, abstractmethod

from rag.schemas import DocumentChunk, SearchResult


class VectorStore(ABC):
    @abstractmethod
    def add(
        self,
        collection: str,
        chunks: list[DocumentChunk],
        vectors: list[list[float]],
    ) -> None:
        """Store chunks and vectors in a collection; existing chunk IDs are replaced."""

    @abstractmethod
    def query(
        self,
        collection: str,
        vector: list[float],
        top_k: int = 5,
        where: dict[str, object] | None = None,
    ) -> list[SearchResult]:
        """Return nearest chunks; `where` restricts by stored metadata."""

    @abstractmethod
    def delete(self, collection: str, where: dict[str, object]) -> None:
        """Remove every entry matching `where` from the collection."""

    def check_ready(self) -> None:
        return None
