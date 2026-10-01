"""
Utility helpers for the backend.
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import Optional


def generate_document_id() -> str:
    """Generate a unique document identifier."""
    return str(uuid.uuid4())


def sanitize_filename(filename: str) -> str:
    """
    Remove potentially dangerous characters from an uploaded filename.

    Returns a safe filename string.
    """
    # Keep only alphanumerics, dots, dashes, underscores
    safe = re.sub(r"[^\w.\-]", "_", filename)
    # Strip leading dots/dashes to prevent hidden files or path traversal
    safe = safe.lstrip(".-")
    return safe or "uploaded_file"


def ensure_upload_dir(upload_dir: str) -> Path:
    """
    Create the upload directory if it does not exist.

    Returns the resolved Path object.
    """
    path = Path(upload_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def file_size_mb(size_bytes: int) -> float:
    """Convert bytes to megabytes."""
    return size_bytes / (1024 * 1024)


def truncate_text(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, appending ellipsis if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def clean_document_id(doc_id: str) -> Optional[str]:
    """
    Validate and return a document_id, or None if it looks invalid.
    Prevents path traversal or injection via document_id parameter.
    """
    if not doc_id:
        return None
    # UUID format: 8-4-4-4-12 hex digits
    if re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", doc_id.lower()):
        return doc_id.lower()
    return None
