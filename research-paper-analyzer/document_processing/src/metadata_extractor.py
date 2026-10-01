"""
Metadata extraction from research paper text.

Attempts to extract:
- Title (from first non-empty lines or PDF metadata)
- Authors
- Year / publication info
- DOI
- Abstract (first-pass)
- Research domain (keyword-based heuristic)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PaperMetadata:
    """Structured metadata extracted from a research paper."""

    title: str = "Unknown Title"
    authors: List[str] = field(default_factory=list)
    year: Optional[str] = None
    doi: Optional[str] = None
    abstract: Optional[str] = None
    journal: Optional[str] = None
    keywords: List[str] = field(default_factory=list)


# ── Regex helpers ────────────────────────────────────────────────────────────

_DOI_PATTERN = re.compile(r"10\.\d{4,}/\S+", re.IGNORECASE)
_YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")
_AUTHOR_SEPARATORS = re.compile(r"[,;]|\band\b", re.IGNORECASE)

# Lines that look like author lists: contains multiple capitalized words
# separated by commas — heuristic only
_AUTHOR_LINE = re.compile(
    r"^([A-Z][a-z]+\.?\s+[A-Z][a-zA-Z\-]+)(\s*[,;&]\s*([A-Z][a-z]+\.?\s+[A-Z][a-zA-Z\-]+))*$"
)


def extract_metadata(
    full_text: str,
    pdf_metadata: Optional[dict] = None,
) -> PaperMetadata:
    """
    Extract metadata from document text and optional PyMuPDF metadata dict.

    Args:
        full_text: Full cleaned text of the paper.
        pdf_metadata: Dict from fitz.Document.metadata (may be empty/None).

    Returns:
        PaperMetadata instance with best-effort extracted values.
    """
    meta = PaperMetadata()

    # ── 1. Use PDF embedded metadata if available ──────────────────────────
    if pdf_metadata:
        raw_title = pdf_metadata.get("title", "").strip()
        if raw_title and len(raw_title) > 3:
            meta.title = raw_title

        raw_authors = pdf_metadata.get("author", "").strip()
        if raw_authors:
            meta.authors = _split_authors(raw_authors)

        raw_subject = pdf_metadata.get("subject", "").strip()
        if raw_subject:
            meta.keywords = [kw.strip() for kw in raw_subject.split(",") if kw.strip()]

    # ── 2. Fallback: heuristic title from first meaningful lines ───────────
    if meta.title == "Unknown Title":
        meta.title = _extract_title_heuristic(full_text)

    # ── 3. Extract DOI ─────────────────────────────────────────────────────
    doi_match = _DOI_PATTERN.search(full_text[:3000])
    if doi_match:
        meta.doi = doi_match.group(0).rstrip(".,)")

    # ── 4. Extract year ────────────────────────────────────────────────────
    year_match = _YEAR_PATTERN.search(full_text[:2000])
    if year_match:
        meta.year = year_match.group(0)

    # ── 5. Extract abstract (first 3000 chars) ─────────────────────────────
    if not meta.abstract:
        meta.abstract = _extract_abstract(full_text)

    logger.info(
        "Metadata extracted — title='%s', authors=%d, doi=%s",
        meta.title[:60],
        len(meta.authors),
        meta.doi,
    )

    return meta


def extract_pdf_metadata(file_path: str) -> dict:
    """
    Read the embedded PDF metadata dict from a file.

    Returns empty dict on failure.
    """
    try:
        import fitz
        doc = fitz.open(file_path)
        metadata = dict(doc.metadata)
        doc.close()
        return metadata
    except Exception as exc:
        logger.warning("Could not read PDF metadata: %s", exc)
        return {}


# ── Private helpers ──────────────────────────────────────────────────────────

def _extract_title_heuristic(text: str) -> str:
    """
    Try to extract a title from the first non-empty lines of the text.
    The title is usually among the first 5 non-empty, non-trivially-short lines.
    """
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    candidates = []
    for line in lines[:10]:
        # Skip lines that look like author lists, DOIs, or page numbers
        if _DOI_PATTERN.search(line):
            continue
        if _YEAR_PATTERN.fullmatch(line):
            continue
        if len(line) < 10 or len(line) > 200:
            continue
        # Skip ALL-CAPS lines (often journal name or conference)
        if line.isupper():
            continue
        candidates.append(line)

    if candidates:
        # The longest of the first 3 candidates is usually the title
        return max(candidates[:3], key=len)

    return "Unknown Title"


def _extract_abstract(text: str) -> Optional[str]:
    """Attempt to find abstract text within the first portion of the document."""
    # Look for explicit Abstract marker
    match = re.search(
        r"(?:^|\n)\s*Abstract[\s:—\-]*\n?([\s\S]{100,2000?})(?=\n\n|\n[A-Z1-9])",
        text[:5000],
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    # Fallback: first large paragraph after the title area
    paragraphs = [p.strip() for p in text[:3000].split("\n\n") if len(p.strip()) > 200]
    if paragraphs:
        return paragraphs[0]

    return None


def _split_authors(author_string: str) -> List[str]:
    """Split an author string into individual names."""
    parts = _AUTHOR_SEPARATORS.split(author_string)
    return [p.strip() for p in parts if p.strip() and len(p.strip()) > 2]
