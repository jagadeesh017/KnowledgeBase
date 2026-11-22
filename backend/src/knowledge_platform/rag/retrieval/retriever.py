from knowledge_platform.rag.embeddings import EmbeddingProvider
from knowledge_platform.rag.schemas import SearchResult
from knowledge_platform.rag.vector_store import VectorStore


class Retriever:
    def __init__(
        self,
        embeddings: EmbeddingProvider,
        store: VectorStore,
        collection: str,
        where: dict[str, object] | None = None,
        max_distance: float | None = None,
    ) -> None:
        self.embeddings = embeddings
        self.store = store
        self.collection = collection
        self.where = where
        self.max_distance = max_distance

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        vector = self.embeddings.embed_one(query)
        results = self.store.query(
            self.collection, vector, top_k=top_k, where=self.where
        )
        if self.max_distance is not None:
            results = [result for result in results if result.distance <= self.max_distance]
        return results
