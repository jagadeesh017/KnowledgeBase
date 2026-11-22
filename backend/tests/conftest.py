from collections.abc import Callable
from pathlib import Path

import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


@pytest.fixture
def make_pdf(tmp_path: Path) -> Callable[..., Path]:
    def _make_pdf(
        filename: str,
        pages: list[str],
        *,
        title: str | None = None,
        author: str | None = None,
    ) -> Path:
        path = tmp_path / filename
        pdf = canvas.Canvas(str(path), pagesize=letter)
        if title is not None:
            pdf.setTitle(title)
        if author is not None:
            pdf.setAuthor(author)

        for text in pages:
            y = 750
            for line in text.splitlines():
                pdf.drawString(72, y, line)
                y -= 14
            pdf.showPage()

        pdf.save()
        return path

    return _make_pdf
