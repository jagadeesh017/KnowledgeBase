from collections.abc import Mapping
from pathlib import Path

from pypdf import PdfReader

from knowledge_platform.rag.schemas.document import LoadedDocument, PageContent


class PDFLoader:
    def load(self, path: str | Path, document_id: str) -> LoadedDocument:
        pdf_path = Path(path)
        reader = PdfReader(str(pdf_path))

        pages: list[PageContent] = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            pages.append(PageContent(page_number=page_number, text=text))

        return LoadedDocument(
            document_id=document_id,
            source=pdf_path.name,
            pages=pages,
            metadata=self._normalise_metadata(reader.metadata),
        )

    @staticmethod
    def _normalise_metadata(metadata: Mapping[str, object] | None) -> dict[str, str]:
        if not metadata:
            return {}
        return {
            str(key).lstrip("/"): str(value)
            for key, value in metadata.items()
            if value is not None
        }
