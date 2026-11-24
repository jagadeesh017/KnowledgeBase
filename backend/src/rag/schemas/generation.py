from pydantic import BaseModel, Field

from rag.schemas.retrieval import Source


class RAGAnswer(BaseModel):
    answer: str
    sources: list[Source] = Field(default_factory=list)
