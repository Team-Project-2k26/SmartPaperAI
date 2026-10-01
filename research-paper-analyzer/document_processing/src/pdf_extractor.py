"""
PDF text extraction using PyMuPDF (fitz).

Extracts text page-by-page, preserves page numbers,
detects image-only/scanned PDFs, and returns a structured
representation of the raw document.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PageData:
    """Holds extracted content for a single PDF page."""

    page_number: int           # 1-indexed
    raw_text: str
    char_count: int
    image_count: int           # number of images on the page
    has_text: bool


@dataclass
class ExtractedDocument:
    """Full structured output from PDF extraction."""

    file_path: str
    total_pages: int
    pages: List[PageData] = field(default_factory=list)
    full_text: str = ""
    is_scanned: bool = False
    extraction_warning: Optional[str] = None

    # Convenience: text grouped by page number
    @property
    def page_texts(self) -> dict[int, str]:
        return {p.page_number: p.raw_text for p in self.pages}


# Threshold: if less than this many characters per page on average,
# the PDF is likely image-based / scanned.
_SCANNED_CHAR_THRESHOLD = 80


def extract_pdf(file_path: str | Path) -> ExtractedDocument:
    """
    Extract text from a PDF file using PyMuPDF.

    Args:
        file_path: Absolute path to the PDF file.

    Returns:
        ExtractedDocument with per-page text and metadata.

    Raises:
        ValueError: If the file is not a valid PDF or cannot be opened.
        FileNotFoundError: If the file does not exist.
    """
    import fitz  # PyMuPDF

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"File is not a PDF: {path.name}")

    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        raise ValueError(f"Cannot open PDF '{path.name}': {exc}") from exc

    if doc.page_count == 0:
        doc.close()
        raise ValueError(f"PDF '{path.name}' contains no pages.")

    pages: List[PageData] = []
    total_chars = 0

    for idx in range(doc.page_count):
        page = doc[idx]
        raw_text = page.get_text("text")  # plain text extraction

        # Count images embedded on this page
        image_list = page.get_images(full=False)
        char_count = len(raw_text.strip())
        total_chars += char_count

        pages.append(
            PageData(
                page_number=idx + 1,
                raw_text=raw_text,
                char_count=char_count,
                image_count=len(image_list),
                has_text=char_count > 10,
            )
        )

    doc.close()

    full_text = "\n".join(p.raw_text for p in pages)

    # Detect scanned/image-only PDFs
    avg_chars_per_page = total_chars / len(pages) if pages else 0
    is_scanned = avg_chars_per_page < _SCANNED_CHAR_THRESHOLD

    warning: Optional[str] = None
    if is_scanned:
        warning = (
            f"This PDF appears to be scanned or image-based "
            f"(avg {avg_chars_per_page:.0f} chars/page). "
            "Text extraction quality may be poor. OCR is not available."
        )
        logger.warning("Scanned PDF detected: %s", path.name)

    logger.info(
        "Extracted %d pages from '%s' (avg %.0f chars/page, scanned=%s)",
        len(pages),
        path.name,
        avg_chars_per_page,
        is_scanned,
    )

    return ExtractedDocument(
        file_path=str(path),
        total_pages=len(pages),
        pages=pages,
        full_text=full_text,
        is_scanned=is_scanned,
        extraction_warning=warning,
    )


def validate_pdf_bytes(content: bytes, filename: str) -> None:
    """
    Validate that raw bytes look like a PDF.

    Args:
        content: Raw file bytes.
        filename: Original filename for error messages.

    Raises:
        ValueError: If content does not appear to be a PDF.
    """
    if not content:
        raise ValueError("Uploaded file is empty.")
    if not content.startswith(b"%PDF"):
        raise ValueError(
            f"'{filename}' does not appear to be a valid PDF file."
        )


def get_pdf_page_count(file_path: str | Path) -> int:
    """Return the total number of pages without full text extraction."""
    import fitz

    doc = fitz.open(str(file_path))
    count = doc.page_count
    doc.close()
    return count
