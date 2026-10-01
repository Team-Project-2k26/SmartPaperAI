"""
QA model loading and caching.

Loads deepset/roberta-base-squad2 as a local extractive QA model.
The model is loaded once per process via a singleton pattern.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_qa_pipeline = None
_QA_MODEL = "deepset/roberta-base-squad2"


def get_qa_pipeline():
    """
    Lazily load and return the QA pipeline singleton.

    Returns:
        HuggingFace question-answering pipeline.
    """
    global _qa_pipeline
    if _qa_pipeline is None:
        logger.info("Loading QA model: %s", _QA_MODEL)
        from transformers import pipeline
        _qa_pipeline = pipeline(
            "question-answering",
            model=_QA_MODEL,
            tokenizer=_QA_MODEL,
        )
        logger.info("QA model loaded successfully.")
    return _qa_pipeline


def is_qa_model_loaded() -> bool:
    """Check whether the QA model has been loaded."""
    return _qa_pipeline is not None
