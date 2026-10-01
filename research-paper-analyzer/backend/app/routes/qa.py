"""
QA route.

POST /chat — Accepts a question + document_id and returns a
             grounded answer from the uploaded paper.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from backend.app.models.schemas import QuestionRequest, QuestionResponse
from backend.app.services.document_service import get_cached_document
from backend.app.services.qa_service import answer_paper_question
from backend.app.utils.helpers import clean_document_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["qa"])


@router.post(
    "",
    response_model=QuestionResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question about the uploaded paper",
    description="Retrieves relevant passages from the paper and returns "
                "an extractive answer grounded in the document.",
)
async def chat_with_paper(body: QuestionRequest) -> QuestionResponse:
    """
    Answer a question about a previously uploaded and analyzed paper.

    If the answer cannot be found in the paper, responds honestly rather
    than hallucinating.
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

    question = body.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty.",
        )

    try:
        response = answer_paper_question(
            question=question,
            processed_doc=processed,
            top_k=5,
        )
    except Exception as exc:
        logger.exception("QA error for doc=%s: %s", doc_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while answering the question.",
        )

    return response
