"""
Text cleaning for extracted PDF content.

Removes common PDF extraction artifacts such as:
- Excessive whitespace / line breaks
- Header/footer repetitions
- Page numbers
- Hyphenated line breaks
- Non-printable characters
- References section (optionally)
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import List

logger = logging.getLogger(__name__)


# ── Regex patterns ──────────────────────────────────────────────────────────

# Hyphenated word break at end of line (e.g. "meth-\nod" → "method")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")

# Multiple blank lines → single blank line
_MULTI_BLANK = re.compile(r"\n{3,}")

# Lone page number on a line (e.g. "\n12\n")
_PAGE_NUMBER_LINE = re.compile(r"^\s*\d{1,4}\s*$", re.MULTILINE)

# Common headers/footers: "Page X of Y", journal name patterns, DOI lines
_HEADER_FOOTER = re.compile(
    r"^.{0,80}(page\s+\d+\s+of\s+\d+|doi\s*:\s*10\.\d+|©\s*\d{4}|all\s+rights\s+reserved).{0,80}$",
    re.IGNORECASE | re.MULTILINE,
)

# Ligatures and common Unicode replacements
_LIGATURE_MAP = str.maketrans(
    {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00ad": "",   # soft hyphen
    }
)


def clean_text(raw_text: str, *, aggressive: bool = False) -> str:
    """
    Clean raw text extracted from a PDF.

    Args:
        raw_text: The raw string returned by PyMuPDF.
        aggressive: If True, apply more aggressive cleaning (remove
                    single-character lines, very short lines likely to be
                    artifacts). Use False for general cleaning.

    Returns:
        Cleaned text string.
    """
    if not raw_text or not raw_text.strip():
        return ""

    text = raw_text

    # 1. Normalize unicode (NFC) and fix ligatures
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_LIGATURE_MAP)

    # 2. Remove non-printable characters (keep newlines/tabs)
    text = "".join(
        ch for ch in text if ch.isprintable() or ch in "\n\t"
    )

    # 3. Fix hyphenated line breaks ("meth-\nod" → "method")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)

    # 4. Remove obvious headers/footers
    text = _HEADER_FOOTER.sub("", text)

    # 5. Remove lone page number lines
    text = _PAGE_NUMBER_LINE.sub("", text)

    # 6. Normalize whitespace within lines
    lines = text.split("\n")
    cleaned_lines: List[str] = []
    for line in lines:
        stripped = " ".join(line.split())   # collapse internal whitespace
        if aggressive and len(stripped) <= 2:
            continue    # skip very short artifact lines
        cleaned_lines.append(stripped)

    # 7. Collapse multiple consecutive blank lines to one
    text = "\n".join(cleaned_lines)
    text = _MULTI_BLANK.sub("\n\n", text)

    return text.strip()


def clean_pages(pages_text: dict[int, str], *, aggressive: bool = False) -> dict[int, str]:
    """
    Apply clean_text to a dict of {page_number: raw_text}.

    Returns:
        Dict with cleaned text per page.
    """
    return {pg: clean_text(raw, aggressive=aggressive) for pg, raw in pages_text.items()}


def remove_references_section(text: str) -> str:
    """
    Attempt to strip the References/Bibliography section from the end of
    the document text, as references are not useful for summarization.

    Returns text up to (but not including) the References header,
    or the original text if no section is found.
    """
    pattern = re.compile(
        r"\n\s*(references|bibliography|works\s+cited)\s*\n",
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if match:
        logger.debug("Stripping References section from position %d", match.start())
        return text[: match.start()].strip()
    return text


def normalize_whitespace(text: str) -> str:
    """Replace any sequence of whitespace chars with a single space."""
    return " ".join(text.split())
