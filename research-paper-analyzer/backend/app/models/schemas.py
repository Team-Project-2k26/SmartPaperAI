"""
Pydantic schemas for all API request and response models.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Upload ────────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    """Response after a successful PDF upload."""

    document_id: str = Field(..., description="UUID identifying this uploaded document")
    filename: str
    total_pages: int
    is_scanned: bool
    warning: Optional[str] = None
    message: str = "PDF uploaded and processed successfully."


# ── Analysis ──────────────────────────────────────────────────────────────────

class FutureResearchItem(BaseModel):
    topic: str
    question: str
    reason: str
    evidence: str
    is_inferred: bool = False


class AnalysisResponse(BaseModel):
    """Full structured analysis of a research paper."""

    document_id: str

    # Paper info
    title: str = ""
    authors: List[str] = []
    year: Optional[str] = None
    domain: str = ""

    # Core analysis
    summary: str = ""
    objective: str = ""
    methodology: str = ""
    dataset: str = ""
    findings: List[str] = []
    main_points: List[str] = []
    contribution: str = ""
    limitations: List[str] = []
    conclusion: str = ""

    # Key insight
    key_insight: str = ""
    simple_explanation: str = ""
    why_it_matters: str = ""

    # Keywords
    keywords: List[str] = []

    # Future research
    future_research: List[FutureResearchItem] = []

    # Meta
    processing_time_seconds: Optional[float] = None


# ── QA ────────────────────────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    """Request body for asking a question about a paper."""

    document_id: str = Field(..., description="The document_id from the upload response")
    question: str = Field(..., min_length=3, max_length=500)


class SourceReference(BaseModel):
    section: str
    pages: List[int] = []
    similarity: float


class QuestionResponse(BaseModel):
    """Response from the QA chatbot."""

    question: str
    answer: str
    found_in_paper: bool
    confidence: float
    sources: List[SourceReference] = []


# ── Error ─────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None
