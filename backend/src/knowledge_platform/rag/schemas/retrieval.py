from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    page_number: int
    distance: float
    metadata: dict[str, str] = Field(default_factory=dict)


class Source(BaseModel):
    index: int
    chunk_id: str
    document_id: str
    source: str
    page_number: int
    distance: float
