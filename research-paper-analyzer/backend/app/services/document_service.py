"""
Document processing service.

Orchestrates the full PDF → structured document pipeline:
  1. Validate and save uploaded PDF
  2. Extract text (PyMuPDF)
  3. Clean text
  4. Detect sections
  5. Extract metadata
  6. Chunk text for retrieval
  7. Compute and cache embeddings

Results are cached in an in-memory store keyed by document_id to
avoid re-processing the same document on follow-up QA requests.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Add project root to sys.path so sibling packages resolve
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from document_processing.src.chunker import TextChunk, chunk_document
from document_processing.src.metadata_extractor import (
    PaperMetadata,
    extract_metadata,
    extract_pdf_metadata,
)
from document_processing.src.pdf_extractor import (
    ExtractedDocument,
    extract_pdf,
    validate_pdf_bytes,
)
from document_processing.src.retriever import embed_chunks
from document_processing.src.section_detector import (
    SegmentedDocument,
    detect_sections,
    extract_abstract_heuristic,
)
from document_processing.src.text_cleaner import (
    clean_text,
    remove_references_section,
)


@dataclass
class ProcessedDocument:
    """Fully processed document stored in the in-memory cache."""

    document_id: str
    filename: str
    extracted: ExtractedDocument
    clean_full_text: str
    segmented: SegmentedDocument
    metadata: PaperMetadata
    chunks: List[TextChunk]
    chunk_embeddings: Optional[np.ndarray] = None

    @property
    def sections_dict(self) -> Dict[str, str]:
        """Return {section_name: text} dict for analysis."""
        return {name: sec.text for name, sec in self.segmented.sections.items()}


# ── In-memory document cache ─────────────────────────────────────────────────
# {document_id: ProcessedDocument}
_document_cache: Dict[str, ProcessedDocument] = {}


def get_cached_document(document_id: str) -> Optional[ProcessedDocument]:
    """Retrieve a previously processed document from cache."""
    return _document_cache.get(document_id)


def process_and_cache(
    document_id: str,
    file_path: str,
    filename: str,
) -> ProcessedDocument:
    """
    Run the full document processing pipeline and cache the result.

    Args:
        document_id: UUID string for this document.
        file_path: Absolute path to the saved PDF file.
        filename: Original uploaded filename.

    Returns:
        ProcessedDocument ready for analysis.

    Raises:
        ValueError: If the PDF cannot be processed.
    """
    if document_id in _document_cache:
        logger.info("Cache hit for document_id=%s", document_id)
        return _document_cache[document_id]

    logger.info("Processing document: %s (id=%s)", filename, document_id)

    # ── 1. Extract text ──────────────────────────────────────────────────────
    extracted = extract_pdf(file_path)
    if not extracted.full_text.strip():
        raise ValueError("No text could be extracted from this PDF.")

    # ── 2. Clean text ────────────────────────────────────────────────────────
    clean_full = clean_text(extracted.full_text, aggressive=False)
    clean_full = remove_references_section(clean_full)

    if len(clean_full.strip()) < 200:
        raise ValueError("The extracted text is too short to analyze. The PDF may be scanned or corrupted.")

    # ── 3. Detect sections ───────────────────────────────────────────────────
    segmented = detect_sections(clean_full)

    # If no sections were detected, try abstract heuristic
    if not segmented.sections:
        abstract = extract_abstract_heuristic(clean_full)
        if abstract:
            from document_processing.src.section_detector import Section
            segmented.sections["abstract"] = Section(
                name="abstract",
                heading="Abstract",
                text=abstract,
                start_line=0,
                end_line=10,
            )

    # ── 4. Extract metadata ──────────────────────────────────────────────────
    pdf_meta_dict = extract_pdf_metadata(file_path)
    metadata = extract_metadata(clean_full, pdf_meta_dict)

    # ── 5. Build clean page map for page attribution ─────────────────────────
    page_map = {p.page_number: clean_text(p.raw_text) for p in extracted.pages}

    # ── 6. Chunk document ────────────────────────────────────────────────────
    sections_dict = {name: sec.text for name, sec in segmented.sections.items()}
    chunks = chunk_document(
        sections=sections_dict,
        full_text=clean_full,
        page_map=page_map,
        chunk_size=5,
        overlap=2,
    )

    # ── 7. Compute embeddings ────────────────────────────────────────────────
    embeddings: Optional[np.ndarray] = None
    if chunks:
        try:
            embeddings = embed_chunks(chunks)
            logger.info("Embeddings computed: shape %s", embeddings.shape)
        except Exception as exc:
            logger.warning("Embedding computation failed: %s", exc)

    processed = ProcessedDocument(
        document_id=document_id,
        filename=filename,
        extracted=extracted,
        clean_full_text=clean_full,
        segmented=segmented,
        metadata=metadata,
        chunks=chunks,
        chunk_embeddings=embeddings,
    )

    _document_cache[document_id] = processed
    logger.info(
        "Document processed and cached: %d sections, %d chunks",
        len(segmented.sections),
        len(chunks),
    )
    return processed


async def save_upload_file(
    file_bytes: bytes,
    filename: str,
    upload_dir: str,
    document_id: str,
) -> str:
    """
    Validate and persist uploaded file bytes to disk.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename: Original filename.
        upload_dir: Directory to save into.
        document_id: UUID used to create a unique file path.

    Returns:
        Absolute path string to the saved file.

    Raises:
        ValueError: If the file is invalid.
    """
    from backend.app.utils.helpers import sanitize_filename, ensure_upload_dir

    validate_pdf_bytes(file_bytes, filename)

    upload_path = ensure_upload_dir(upload_dir)
    safe_name = sanitize_filename(filename)
    dest = upload_path / f"{document_id}_{safe_name}"

    with open(dest, "wb") as f:
        f.write(file_bytes)

    logger.info("Saved uploaded file to %s (%d bytes)", dest, len(file_bytes))
    return str(dest)
