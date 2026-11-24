from pathlib import Path
from typing import Any, cast

import chromadb
from chromadb.api import ClientAPI
from chromadb.errors import NotFoundError

from knowledge_platform.exceptions import ServiceUnavailableError
from knowledge_platform.rag.schemas import DocumentChunk, SearchResult
from knowledge_platform.rag.vector_store.base import VectorStore


class ChromaVectorStore(VectorStore):
    DEFAULT_PATH = "./chroma"

    def __init__(
        self,
        path: str | Path = DEFAULT_PATH,
        client: ClientAPI | None = None,
    ) -> None:
        self.client = client if client is not None else chromadb.PersistentClient(path=str(path))

    def add(
        self,
        collection: str,
        chunks: list[DocumentChunk],
        vectors: list[list[float]],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return

        target = self.client.get_or_create_collection(name=collection)
        target.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=cast(Any, [list(vector) for vector in vectors]),
            documents=[chunk.content for chunk in chunks],
            metadatas=[self._metadata(chunk) for chunk in chunks],
        )

    def query(
        self,
        collection: str,
        vector: list[float],
        top_k: int = 5,
        where: dict[str, object] | None = None,
    ) -> list[SearchResult]:
        try:
            target = self.client.get_collection(name=collection)
        except NotFoundError:
            return []

        result = target.query(
            query_embeddings=cast(Any, [list(vector)]),
            n_results=top_k,
            where=cast(Any, where),
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"] or []
        documents = result["documents"] or []
        metadatas = result["metadatas"] or []
        distances = result["distances"] or []
        return [
            SearchResult(
                chunk_id=chunk_id,
                document_id=str(metadata["document_id"]),
                content=content,
                page_number=int(str(metadata["page_number"])),
                distance=float(distance),
                metadata={
                    key: str(value)
                    for key, value in metadata.items()
                    if key not in {"document_id", "page_number"}
                },
            )
            for chunk_id, content, metadata, distance in zip(
                ids[0], documents[0], metadatas[0], distances[0]
            )
        ]

    def delete(self, collection: str, where: dict[str, object]) -> None:
        try:
            target = self.client.get_collection(name=collection)
        except NotFoundError:
            return
        target.delete(where=cast(Any, where))

    def check_ready(self) -> None:
        try:
            self.client.list_collections()
        except Exception as error:
            raise ServiceUnavailableError(
                f"Chroma is unavailable: {error}"
            ) from error

    @staticmethod
    def _metadata(chunk: DocumentChunk) -> dict[str, str | int]:
        return {
            "document_id": chunk.document_id,
            "page_number": chunk.page_number,
            **chunk.metadata,
        }
