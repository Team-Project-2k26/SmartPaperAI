"""
Tests for QA retrieval and the /chat endpoint.

Tests context selection, QA inference fallback behavior,
and the chat API endpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from document_processing.src.chunker import chunk_text, TextChunk
from qa_model.src.model.inference import answer_question, _not_found


# ── Chunker tests ─────────────────────────────────────────────────────────────

class TestChunker:
    def test_chunk_basic(self):
        text = (
            "The first sentence describes the method. "
            "The second sentence provides experimental details. "
            "The third sentence discusses the results obtained. "
            "The fourth sentence draws conclusions from findings. "
            "The fifth sentence outlines future work directions. "
            "The sixth sentence summarizes the key contributions."
        )
        chunks = chunk_text(text, chunk_size=3, overlap=1)
        assert len(chunks) >= 1
        assert all(isinstance(c, TextChunk) for c in chunks)

    def test_chunk_empty_text(self):
        chunks = chunk_text("", chunk_size=5, overlap=2)
        assert chunks == []

    def test_chunk_overlap_creates_more_chunks(self):
        text = " ".join([f"Sentence number {i} with some content here." for i in range(10)])
        chunks_no_overlap = chunk_text(text, chunk_size=3, overlap=0)
        chunks_with_overlap = chunk_text(text, chunk_size=3, overlap=1)
        assert len(chunks_with_overlap) >= len(chunks_no_overlap)

    def test_chunk_ids_unique(self):
        text = " ".join([f"Sentence {i} with content." for i in range(15)])
        chunks = chunk_text(text, chunk_size=3, overlap=1)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))


# ── QA inference tests ────────────────────────────────────────────────────────

class TestQAInference:
    def test_empty_question_returns_not_found(self):
        result = answer_question("", "Some context.")
        assert result["found_in_paper"] is False

    def test_empty_context_returns_not_found(self):
        result = answer_question("What is the method?", "")
        assert result["found_in_paper"] is False

    def test_not_found_helper(self):
        result = _not_found("Custom message.")
        assert result["found_in_paper"] is False
        assert result["score"] == 0.0
        assert "Custom message" in result["answer"]

    @pytest.mark.slow
    def test_qa_with_real_context(self):
        """
        Integration test: runs actual QA model inference.
        Marked slow — only runs when model is available.
        """
        context = (
            "The authors propose a BERT-based model for text classification. "
            "The dataset used is IMDb movie reviews with 25,000 training samples. "
            "The model achieves 94.2% accuracy on the test set."
        )
        question = "What dataset was used?"
        result = answer_question(question, context)
        # If model is available, it should find something
        # If not, it should fail gracefully
        assert isinstance(result["answer"], str)
        assert isinstance(result["score"], float)


# ── Chat API endpoint tests ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_chat_with_invalid_document_id():
    """Chat with a non-existent document_id should return 404."""
    from httpx import AsyncClient, ASGITransport
    from backend.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/chat",
            json={
                "document_id": "00000000-0000-0000-0000-000000000000",
                "question": "What is this paper about?",
            },
        )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_chat_with_malformed_document_id():
    """Chat with a malformed document_id should return 400."""
    from httpx import AsyncClient, ASGITransport
    from backend.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/chat",
            json={
                "document_id": "not-a-valid-uuid",
                "question": "What is this paper about?",
            },
        )
    assert response.status_code in (400, 422)


@pytest.mark.asyncio
async def test_chat_with_empty_question():
    """Chat with empty question should return 422 (Pydantic validation)."""
    from httpx import AsyncClient, ASGITransport
    from backend.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/chat",
            json={
                "document_id": "00000000-0000-0000-0000-000000000000",
                "question": "ab",  # too short (min_length=3)
            },
        )
    assert response.status_code == 422
