"""
Summarization service.

Bridges the FastAPI layer and the summarization inference pipeline.
Converts internal PaperAnalysis dataclass into the API's AnalysisResponse schema.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from backend.app.models.schemas import AnalysisResponse, FutureResearchItem
from backend.app.services.document_service import ProcessedDocument
from summarization.src.inference import PaperAnalysis, analyze_paper

logger = logging.getLogger(__name__)


def run_analysis(
    processed_doc: ProcessedDocument,
    use_abstractive: bool = True,
) -> AnalysisResponse:
    """
    Run the full NLP analysis pipeline on a processed document.

    Args:
        processed_doc: A ProcessedDocument from document_service.
        use_abstractive: Whether to use DistilBART for abstractive summary.

    Returns:
        AnalysisResponse Pydantic model ready to return from the API.
    """
    start = time.time()

    logger.info("Starting analysis for document_id=%s", processed_doc.document_id)

    # Run the analysis pipeline
    analysis: PaperAnalysis = analyze_paper(
        sections=processed_doc.sections_dict,
        full_text=processed_doc.clean_full_text,
        metadata=processed_doc.metadata,
        use_abstractive=use_abstractive,
    )

    elapsed = round(time.time() - start, 2)

    # Convert future_research dataclasses to Pydantic models
    future_research_items = [
        FutureResearchItem(
            topic=fr.topic,
            question=fr.question,
            reason=fr.reason,
            evidence=fr.evidence,
            is_inferred=fr.is_inferred,
        )
        for fr in analysis.future_research
    ]

    response = AnalysisResponse(
        document_id=processed_doc.document_id,
        title=analysis.title,
        authors=analysis.authors,
        year=analysis.year,
        domain=analysis.domain,
        summary=analysis.summary,
        objective=analysis.objective,
        methodology=analysis.methodology,
        dataset=analysis.dataset,
        findings=analysis.findings,
        main_points=analysis.main_points,
        contribution=analysis.contribution,
        limitations=analysis.limitations,
        conclusion=analysis.conclusion,
        key_insight=analysis.key_insight,
        simple_explanation=analysis.simple_explanation,
        why_it_matters=analysis.why_it_matters,
        keywords=analysis.keywords,
        future_research=future_research_items,
        processing_time_seconds=elapsed,
    )

    logger.info(
        "Analysis complete for '%s' in %.1fs",
        analysis.title[:60],
        elapsed,
    )

    return response
