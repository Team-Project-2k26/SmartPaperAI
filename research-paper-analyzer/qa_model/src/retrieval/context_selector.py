"""
Context selector for QA retrieval.

Orchestrates:
1. Query embedding
2. Chunk retrieval via cosine similarity
3. Context string assembly with source references
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from document_processing.src.chunker import TextChunk
from document_processing.src.retriever import (
    build_context_string,
    embed_query,
    retrieve_relevant_chunks,
)

logger = logging.getLogger(__name__)


def select_context_for_question(
    question: str,
    chunks: List[TextChunk],
    chunk_embeddings: np.ndarray,
    top_k: int = 5,
    max_context_tokens: int = 1500,
) -> Tuple[str, List[dict]]:
    """
    Select the most relevant context chunks for a given question.

    Args:
        question: User's question string.
        chunks: All document chunks from the paper.
        chunk_embeddings: Pre-computed embeddings for chunks.
        top_k: Number of top chunks to retrieve.
        max_context_tokens: Approximate token limit for assembled context.

    Returns:
        Tuple of (context_string, source_references)
        - context_string: Text to pass to QA model
        - source_references: List of {section, pages, similarity} dicts
    """
    if not chunks or chunk_embeddings is None or len(chunk_embeddings) == 0:
        logger.warning("No chunks/embeddings available for retrieval.")
        return "", []

    retrieved = retrieve_relevant_chunks(
        query=question,
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        top_k=top_k,
    )

    if not retrieved:
        logger.warning("No relevant chunks found for question: %s", question[:80])
        return "", []

    context_str, sources = build_context_string(retrieved, max_tokens=max_context_tokens)

    logger.debug(
        "Context assembled: %d chars from %d chunks",
        len(context_str),
        len(retrieved),
    )

    return context_str, sources
