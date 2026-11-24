from collections.abc import Mapping
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

from knowledge_platform.exceptions import DocumentParseError
from knowledge_platform.rag.schemas.document import LoadedDocument, PageContent


class PDFLoader:
    def load(self, path: str | Path, document_id: str) -> LoadedDocument:
        pdf_path = Path(path)
        return self.load_bytes(pdf_path.read_bytes(), document_id, source=pdf_path.name)

    def load_bytes(
        self,
        data: bytes,
        document_id: str,
        *,
        source: str = "document.pdf",
    ) -> LoadedDocument:
        if not data:
            raise DocumentParseError("PDF file is empty")

        try:
            reader = PdfReader(BytesIO(data))
            pages: list[PageContent] = []
            for page_number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if not text:
                    continue
                pages.append(PageContent(page_number=page_number, text=text))
            metadata = self._normalise_metadata(reader.metadata)
        except Exception as error:
            raise DocumentParseError(f"Could not read PDF: {error}") from error

        return LoadedDocument(
            document_id=document_id,
            source=source,
            pages=pages,
            metadata=metadata,
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
