"""
Analysis route.

POST /analyze — Runs the full NLP analysis pipeline on an uploaded
               document identified by document_id.

Returns structured analysis: summary, findings, key insight,
future research directions, etc.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from backend.app.config.settings import settings
from backend.app.models.schemas import AnalysisResponse
from backend.app.services.document_service import get_cached_document
from backend.app.services.summarization_service import run_analysis
from backend.app.utils.helpers import clean_document_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["analysis"])


class AnalyzeRequest(BaseModel):
    document_id: str


@router.post(
    "",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze an uploaded research paper",
    description="Runs the full NLP pipeline: summary, objective, methodology, "
                "findings, key insight, keywords, and future research directions.",
)
async def analyze_document(body: AnalyzeRequest) -> AnalysisResponse:
    """
    Analyze a previously uploaded document.

    Requires:
        document_id returned from POST /upload
    """
    doc_id = clean_document_id(body.document_id)
    if not doc_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document_id format.",
        )

    processed = get_cached_document(doc_id)
    if processed is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found. Please upload the PDF first.",
        )

    try:
        result = run_analysis(
            processed_doc=processed,
            use_abstractive=settings.use_abstractive_summary,
        )
    except Exception as exc:
        logger.exception("Analysis pipeline error for doc=%s: %s", doc_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Analysis failed. Please try again or use a different PDF.",
        )

    return result
