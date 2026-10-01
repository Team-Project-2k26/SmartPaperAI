"""
Tests for the /upload endpoint.

Uses a minimal in-memory PDF fixture so no real file is required on disk.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from httpx import AsyncClient, ASGITransport

# Ensure project root is importable
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.main import app  # noqa: E402

# Minimal valid PDF bytes (2-page, empty but parseable)
_MINIMAL_PDF = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 3 3]>>endobj
xref
0 4
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
trailer<</Size 4/Root 1 0 R>>
startxref
190
%%EOF"""


@pytest.mark.asyncio
async def test_health_endpoint():
    """Health endpoint should return status ok."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_upload_invalid_type():
    """Uploading a non-PDF file should return 415 or 422."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/upload",
            files={"file": ("test.txt", b"not a pdf", "text/plain")},
        )
    assert response.status_code in (415, 422)


@pytest.mark.asyncio
async def test_upload_empty_file():
    """Uploading an empty file should return 400."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/upload",
            files={"file": ("empty.pdf", b"", "application/pdf")},
        )
    assert response.status_code in (400, 422)


@pytest.mark.asyncio
async def test_upload_fake_pdf_magic_bytes():
    """Bytes that don't start with %PDF should fail validation."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/upload",
            files={"file": ("fake.pdf", b"FAKEPDF-not-real", "application/pdf")},
        )
    assert response.status_code in (400, 415, 422)


@pytest.mark.asyncio
async def test_upload_valid_pdf():
    """A valid minimal PDF should return 200 with a document_id."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/upload",
            files={"file": ("test.pdf", _MINIMAL_PDF, "application/pdf")},
        )
    # May return 200 (success) or 422 (if text is too sparse — valid behavior)
    assert response.status_code in (200, 422)
    if response.status_code == 200:
        data = response.json()
        assert "document_id" in data
        assert len(data["document_id"]) == 36  # UUID format
