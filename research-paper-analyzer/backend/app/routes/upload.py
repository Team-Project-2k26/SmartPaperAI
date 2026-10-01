"""
Upload route.

POST /upload — Accepts a PDF file, validates it, saves it, and runs
               the document processing pipeline.

Returns a document_id that is used in subsequent /analyze and /chat calls.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from backend.app.config.settings import settings
from backend.app.models.schemas import UploadResponse
from backend.app.services.document_service import process_and_cache, save_upload_file
from backend.app.utils.helpers import file_size_mb, generate_document_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/upload", tags=["upload"])


@router.post(
    "",
    response_model=UploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload a research paper PDF",
    description="Upload a PDF file. The backend will extract text, detect sections, "
                "and compute embeddings. Returns a document_id for subsequent calls.",
)
async def upload_pdf(
    file: UploadFile = File(..., description="PDF file to analyze"),
) -> UploadResponse:
    """
    Accept and process an uploaded PDF research paper.

    Validates:
    - File must be a PDF (MIME type + magic bytes)
    - File must not exceed MAX_UPLOAD_SIZE_MB
    """
    # ── Validate content type ────────────────────────────────────────────────
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Only PDF files are accepted. Got: {file.content_type}",
        )

    # ── Read file bytes ──────────────────────────────────────────────────────
    file_bytes = await file.read()

    # ── Validate file size ───────────────────────────────────────────────────
    size_mb = file_size_mb(len(file_bytes))
    if size_mb > settings.max_upload_size_mb:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size {size_mb:.1f} MB exceeds limit of {settings.max_upload_size_mb} MB.",
        )

    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    document_id = generate_document_id()
    filename = file.filename or "uploaded.pdf"

    try:
        # ── Save file ────────────────────────────────────────────────────────
        file_path = await save_upload_file(
            file_bytes=file_bytes,
            filename=filename,
            upload_dir=settings.upload_dir,
            document_id=document_id,
        )

        # ── Process document ─────────────────────────────────────────────────
        processed = process_and_cache(
            document_id=document_id,
            file_path=file_path,
            filename=filename,
        )

    except ValueError as exc:
        logger.warning("Upload validation error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Unexpected error during upload: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing the uploaded file.",
        )

    return UploadResponse(
        document_id=document_id,
        filename=filename,
        total_pages=processed.extracted.total_pages,
        is_scanned=processed.extracted.is_scanned,
        warning=processed.extracted.extraction_warning,
        message="PDF uploaded and processed successfully.",
    )
