"""
FastAPI application entry point.

Registers all routers, configures CORS, and sets up model pre-loading
on startup so the first request doesn't pay the model-loading cost.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# ── Ensure project root is on sys.path ──────────────────────────────────────
# This allows all sibling packages (document_processing, summarization,
# qa_model) to be importable when running: uvicorn backend.main:app
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config.settings import settings
from backend.app.routes.analysis import router as analysis_router
from backend.app.routes.qa import router as qa_router
from backend.app.routes.upload import router as upload_router

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager

# ── Lifespan lifecycle ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Warm up ML models on startup so the first user request is fast.
    Models are loaded lazily inside their respective modules, but we
    trigger them here to front-load the waiting time.
    """
    logger.info("Preloading NLP models...")

    try:
        from document_processing.src.retriever import _get_model
        _get_model()
        logger.info("Sentence-transformer model ready.")
    except Exception as exc:
        logger.warning("Could not preload sentence-transformer: %s", exc)

    try:
        from qa_model.src.model.load_model import get_qa_pipeline
        get_qa_pipeline()
        logger.info("QA model ready.")
    except Exception as exc:
        logger.warning("Could not preload QA model: %s", exc)

    try:
        from summarization.src.inference import _get_summarizer
        _get_summarizer()
        logger.info("Summarization model ready.")
    except Exception as exc:
        logger.warning("Could not preload summarization model: %s", exc)

    logger.info("Server initialization complete.")
    yield


# ── App factory ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="SmartPaperAI — Research Paper Analyzer API",
    description=(
        "AI-powered research paper analysis using open-source NLP models. "
        "Upload a PDF and receive structured summaries, key insights, "
        "future research directions, and a grounded Q&A chatbot."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────────────────
app.include_router(upload_router)
app.include_router(analysis_router)
app.include_router(qa_router)


# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["health"], summary="Health check")
def health_check() -> dict:
    """Returns server status and loaded model info."""
    from qa_model.src.model.load_model import is_qa_model_loaded

    return {
        "status": "ok",
        "version": "1.0.0",
        "models": {
            "qa": "loaded" if is_qa_model_loaded() else "not_loaded",
        },
    }
