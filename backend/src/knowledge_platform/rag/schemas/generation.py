from pydantic import BaseModel, Field

from knowledge_platform.rag.schemas.retrieval import Source


class RAGAnswer(BaseModel):
    answer: str
    sources: list[Source] = Field(default_factory=list)
