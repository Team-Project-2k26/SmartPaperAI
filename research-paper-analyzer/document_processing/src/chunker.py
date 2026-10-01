"""
Text chunking for semantic retrieval.

Splits cleaned document text into overlapping chunks that are
suitable for embedding-based retrieval (used by the QA chatbot).

Also maintains page-tracking so each chunk can report which page(s)
it came from.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

import nltk

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """A chunk of text with positional metadata."""

    chunk_id: int
    text: str
    sentences: List[str] = field(default_factory=list)
    page_numbers: List[int] = field(default_factory=list)   # pages this chunk spans
    section_name: Optional[str] = None                       # if chunk came from a named section
    char_start: int = 0
    char_end: int = 0


def chunk_text(
    text: str,
    chunk_size: int = 5,          # number of sentences per chunk
    overlap: int = 2,             # sentence overlap between adjacent chunks
    section_name: Optional[str] = None,
    page_map: Optional[dict] = None,   # {page_number: text} to find page origins
) -> List[TextChunk]:
    """
    Split text into overlapping sentence-level chunks.

    Args:
        text: Cleaned section or full document text.
        chunk_size: How many sentences per chunk.
        overlap: How many sentences to overlap between consecutive chunks.
        section_name: Optional label for which section these chunks came from.
        page_map: Optional {page_num: page_text} used to assign page numbers.

    Returns:
        List of TextChunk objects.
    """
    if not text or not text.strip():
        return []

    # Tokenize into sentences
    try:
        sentences = nltk.sent_tokenize(text)
    except Exception:
        # Fallback: split on ". " pattern
        sentences = re.split(r"(?<=[.!?])\s+", text)

    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if not sentences:
        return []

    chunks: List[TextChunk] = []
    step = max(1, chunk_size - overlap)
    chunk_id = 0

    for start in range(0, len(sentences), step):
        end = min(start + chunk_size, len(sentences))
        chunk_sents = sentences[start:end]
        chunk_text_str = " ".join(chunk_sents)

        chunk = TextChunk(
            chunk_id=chunk_id,
            text=chunk_text_str,
            sentences=chunk_sents,
            section_name=section_name,
        )

        # Assign page numbers if page_map provided
        if page_map:
            pages = _find_pages_for_chunk(chunk_text_str, page_map)
            chunk.page_numbers = pages

        chunks.append(chunk)
        chunk_id += 1

        if end == len(sentences):
            break

    logger.debug(
        "Chunked %d sentences → %d chunks (size=%d, overlap=%d, section=%s)",
        len(sentences),
        len(chunks),
        chunk_size,
        overlap,
        section_name,
    )

    return chunks


def chunk_document(
    sections: dict[str, str],
    full_text: str,
    page_map: Optional[dict] = None,
    chunk_size: int = 5,
    overlap: int = 2,
) -> List[TextChunk]:
    """
    Chunk an entire document, section by section.

    Args:
        sections: {section_name: section_text} from SegmentedDocument.
        full_text: Fallback full document text if sections is empty.
        page_map: Optional {page_num: text} for page attribution.
        chunk_size: Sentences per chunk.
        overlap: Sentence overlap.

    Returns:
        Flat list of all TextChunk objects.
    """
    all_chunks: List[TextChunk] = []
    chunk_id_offset = 0

    if sections:
        for section_name, section_text in sections.items():
            if not section_text or section_name == "references":
                continue
            chunks = chunk_text(
                section_text,
                chunk_size=chunk_size,
                overlap=overlap,
                section_name=section_name,
                page_map=page_map,
            )
            # Offset chunk IDs to keep them globally unique
            for chunk in chunks:
                chunk.chunk_id += chunk_id_offset
            all_chunks.extend(chunks)
            chunk_id_offset += len(chunks)
    else:
        # No section segmentation — chunk the full text
        chunks = chunk_text(
            full_text,
            chunk_size=chunk_size,
            overlap=overlap,
            section_name="full_document",
            page_map=page_map,
        )
        all_chunks.extend(chunks)

    logger.info("Total document chunks produced: %d", len(all_chunks))
    return all_chunks


# ── Helpers ──────────────────────────────────────────────────────────────────

def _find_pages_for_chunk(chunk_text: str, page_map: dict) -> List[int]:
    """
    Find which page(s) a chunk of text originated from.
    Uses simple substring matching (first 60 chars of chunk).
    """
    sample = chunk_text[:60].strip()
    matching_pages = []
    for page_num, page_text in page_map.items():
        if sample and sample in page_text:
            matching_pages.append(page_num)
    return matching_pages or []
