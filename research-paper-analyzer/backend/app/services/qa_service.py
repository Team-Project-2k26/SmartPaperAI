"""
QA service.

Orchestrates retrieval + inference for the question-answering chatbot.
"""

from __future__ import annotations

import logging
from typing import Optional

from backend.app.models.schemas import QuestionResponse, SourceReference
from backend.app.services.document_service import ProcessedDocument
from qa_model.src.model.inference import answer_question
from qa_model.src.retrieval.context_selector import select_context_for_question

logger = logging.getLogger(__name__)


def answer_paper_question(
    question: str,
    processed_doc: ProcessedDocument,
    top_k: int = 5,
) -> QuestionResponse:
    """
    Answer a user question grounded in the uploaded paper.

    Pipeline:
      1. Retrieve relevant document chunks via semantic similarity
      2. Assemble context string
      3. Run extractive QA model
      4. Return structured response with source references

    Args:
        question: User's question string.
        processed_doc: The ProcessedDocument from the document cache.
        top_k: Number of top chunks to retrieve for context.

    Returns:
        QuestionResponse with answer, confidence, and source info.
    """
    logger.info(
        "QA request for doc=%s | question='%s'",
        processed_doc.document_id,
        question[:80],
    )

    # ── 1. Retrieve context ──────────────────────────────────────────────────
    if not processed_doc.chunks or processed_doc.chunk_embeddings is None:
        # No embeddings — use full text as fallback context
        context_str = processed_doc.clean_full_text[:8000]
        sources = []
        logger.warning("No embeddings available; using raw text fallback for QA.")
    else:
        context_str, raw_sources = select_context_for_question(
            question=question,
            chunks=processed_doc.chunks,
            chunk_embeddings=processed_doc.chunk_embeddings,
            top_k=top_k,
        )
        sources = [
            SourceReference(
                section=s["section"],
                pages=s["pages"],
                similarity=s["similarity"],
            )
            for s in raw_sources
        ]

    if not context_str:
        return QuestionResponse(
            question=question,
            answer="I could not find relevant content in the uploaded paper to answer this question.",
            found_in_paper=False,
            confidence=0.0,
            sources=[],
        )

    # ── 2. Run QA model ──────────────────────────────────────────────────────
    qa_result = answer_question(question=question, context=context_str)

    return QuestionResponse(
        question=question,
        answer=qa_result["answer"],
        found_in_paper=qa_result["found_in_paper"],
        confidence=qa_result["score"],
        sources=sources,
    )
