import pytest

from knowledge_platform.rag.loaders import PDFLoader


def test_extracts_text_page_by_page(make_pdf) -> None:
    path = make_pdf("rag_guide.pdf", ["Introduction to RAG.", "Second page here."])

    document = PDFLoader().load(path, document_id="doc_001")

    assert document.document_id == "doc_001"
    assert document.source == "rag_guide.pdf"
    assert [page.page_number for page in document.pages] == [1, 2]
    assert [page.text for page in document.pages] == [
        "Introduction to RAG.",
        "Second page here.",
    ]


def test_empty_pages_are_skipped(make_pdf) -> None:
    path = make_pdf("gaps.pdf", ["Alpha", "", "Gamma"])

    document = PDFLoader().load(path, document_id="doc_002")

    assert [page.page_number for page in document.pages] == [1, 3]
    assert [page.text for page in document.pages] == ["Alpha", "Gamma"]


def test_document_metadata_is_preserved(make_pdf) -> None:
    path = make_pdf("meta.pdf", ["Body text"], title="RAG Guide", author="Ada")

    document = PDFLoader().load(path, document_id="doc_003")

    assert document.metadata["Title"] == "RAG Guide"
    assert document.metadata["Author"] == "Ada"


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        PDFLoader().load("does_not_exist.pdf", document_id="doc_004")
