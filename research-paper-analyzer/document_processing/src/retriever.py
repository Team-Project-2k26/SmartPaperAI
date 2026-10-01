"""
Semantic retrieval for QA context selection.

Uses sentence-transformers (all-MiniLM-L6-v2) to embed document
chunks and retrieve the most relevant ones for a given query using
cosine similarity.

Model is loaded once and reused across calls.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from document_processing.src.chunker import TextChunk

logger = logging.getLogger(__name__)

# Singleton: loaded once per process
_embedding_model = None
_MODEL_NAME = "all-MiniLM-L6-v2"


def _get_model():
    """Lazy-load the sentence transformer model."""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading sentence-transformer model: %s", _MODEL_NAME)
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(_MODEL_NAME)
        logger.info("Sentence-transformer model loaded.")
    return _embedding_model


def embed_chunks(chunks: List[TextChunk]) -> np.ndarray:
    """
    Compute embeddings for a list of TextChunks.

    Args:
        chunks: List of TextChunk objects.

    Returns:
        numpy array of shape (N, embedding_dim).
    """
    model = _get_model()
    texts = [chunk.text for chunk in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """
    Compute the embedding for a query string.

    Returns:
        1-D numpy array (embedding_dim,).
    """
    model = _get_model()
    return model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]


def retrieve_relevant_chunks(
    query: str,
    chunks: List[TextChunk],
    chunk_embeddings: np.ndarray,
    top_k: int = 5,
) -> List[Tuple[TextChunk, float]]:
    """
    Retrieve the top-k most relevant chunks for a query.

    Args:
        query: User question or query string.
        chunks: List of TextChunk objects corresponding to chunk_embeddings.
        chunk_embeddings: Pre-computed embeddings (N, dim) from embed_chunks().
        top_k: Number of top chunks to return.

    Returns:
        List of (TextChunk, similarity_score) sorted by descending similarity.
    """
    if not chunks or chunk_embeddings is None or len(chunk_embeddings) == 0:
        return []

    query_vec = embed_query(query)

    # Cosine similarity: dot product of normalised vectors
    norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    normalised_chunks = chunk_embeddings / norms

    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return []
    normalised_query = query_vec / query_norm

    scores = normalised_chunks @ normalised_query   # shape (N,)

    k = min(top_k, len(chunks))
    top_indices = np.argsort(scores)[::-1][:k]

    results = [
        (chunks[idx], float(scores[idx]))
        for idx in top_indices
        if scores[idx] > 0.1   # filter out near-zero similarity
    ]

    logger.debug(
        "Retrieved %d relevant chunks for query (top score=%.3f)",
        len(results),
        results[0][1] if results else 0,
    )

    return results


def build_context_string(
    retrieved: List[Tuple[TextChunk, float]],
    max_tokens: int = 1500,
) -> Tuple[str, List[dict]]:
    """
    Build a context string from retrieved chunks for QA model input.

    Args:
        retrieved: List of (TextChunk, score) from retrieve_relevant_chunks.
        max_tokens: Approximate character limit (1 token ≈ 4 chars).

    Returns:
        Tuple of (context_string, source_references)
        where source_references is a list of dicts with page/section info.
    """
    max_chars = max_tokens * 4
    context_parts = []
    sources = []
    total_chars = 0

    for chunk, score in retrieved:
        text = chunk.text.strip()
        if total_chars + len(text) > max_chars:
            break
        context_parts.append(text)
        total_chars += len(text)

        sources.append(
            {
                "section": chunk.section_name or "unknown",
                "pages": chunk.page_numbers,
                "similarity": round(score, 3),
            }
        )

    return "\n\n".join(context_parts), sources
