from pydantic import BaseModel, Field


class PageContent(BaseModel):
    page_number: int = Field(gt=0, description="1-based page number in the source PDF")
    text: str = Field(description="Text extracted from this page")
    metadata: dict[str, str] = Field(default_factory=dict)


class LoadedDocument(BaseModel):
    document_id: str = Field(description="Caller-supplied identifier for the document")
    source: str = Field(description="Original filename of the PDF")
    pages: list[PageContent] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    chunk_id: str = Field(description="Sequential identifier, e.g. chunk_001")
    document_id: str
    content: str
    page_number: int = Field(gt=0)
    metadata: dict[str, str] = Field(default_factory=dict)
