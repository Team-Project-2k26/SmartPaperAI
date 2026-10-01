"""
QA inference module.

Runs extractive question answering using roberta-base-squad2 on
a provided context string. If the model cannot find a confident
answer, returns a fallback message rather than hallucinating.
"""

from __future__ import annotations

import logging
from typing import Optional

from qa_model.src.model.load_model import get_qa_pipeline

logger = logging.getLogger(__name__)

# Minimum confidence score to return an answer
_MIN_SCORE_THRESHOLD = 0.05


def answer_question(
    question: str,
    context: str,
) -> dict:
    """
    Run extractive QA on the provided context.

    Args:
        question: The user's question about the paper.
        context: Retrieved relevant passages from the document.

    Returns:
        Dict with keys:
          - answer (str): extracted answer or fallback message
          - score (float): model confidence score
          - found_in_paper (bool): whether answer was found
          - start (int): character start in context
          - end (int): character end in context
    """
    if not question or not question.strip():
        return _not_found("Empty question received.")

    if not context or not context.strip():
        return _not_found("No relevant context found in the uploaded paper.")

    # Truncate context to ~3800 tokens (approx 15000 chars)
    context = context[:15000]

    try:
        qa_pipeline = get_qa_pipeline()
        result = qa_pipeline(
            question=question,
            context=context,
            max_answer_len=300,
            handle_impossible_answer=True,
        )

        answer_text = result.get("answer", "").strip()
        score = float(result.get("score", 0.0))

        # RoBERTa returns empty string for "impossible" answers
        if not answer_text or answer_text == "":
            return _not_found("I could not find enough information about this in the uploaded paper.")

        if score < _MIN_SCORE_THRESHOLD:
            return _not_found(
                "I could not find a confident answer to this question in the uploaded paper."
            )

        logger.debug("QA answer: '%s' (score=%.3f)", answer_text[:80], score)

        return {
            "answer": answer_text,
            "score": round(score, 4),
            "found_in_paper": True,
            "start": result.get("start", 0),
            "end": result.get("end", 0),
        }

    except Exception as exc:
        logger.warning("Neural QA inference unavailable (%s); falling back to extractive matching.", exc)
        return _extractive_fallback_qa(question=question, context=context)


def _extractive_fallback_qa(question: str, context: str) -> dict:
    """
    Extractive sentence ranking fallback when neural QA pipeline is unavailable.
    Finds the most relevant grounded sentence in context based on keyword overlap.
    """
    import re

    if not context or not question:
        return _not_found("I could not find enough information in the uploaded paper.")

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context) if len(s.strip()) > 15]
    if not sentences:
        return _not_found("I could not find enough information in the uploaded paper.")

    q_words = set(re.findall(r"\w+", question.lower()))
    stopwords = {
        "what", "is", "the", "of", "this", "paper", "how", "why", "in", "to", "a", "an",
        "does", "do", "and", "or", "for", "are", "tell", "me", "about", "can", "you",
        "which", "who", "whom", "where", "when", "there"
    }
    content_words = q_words - stopwords
    if not content_words:
        content_words = q_words

    best_sent = ""
    best_score = 0.0

    for sent in sentences:
        s_words = set(re.findall(r"\w+", sent.lower()))
        if not s_words:
            continue
        overlap = len(content_words & s_words)
        score = overlap / (len(content_words) + 0.1)
        if score > best_score:
            best_score = score
            best_sent = sent

    if best_score > 0.2 and best_sent:
        start_idx = context.find(best_sent)
        return {
            "answer": best_sent,
            "score": round(float(min(0.95, best_score)), 4),
            "found_in_paper": True,
            "start": max(0, start_idx),
            "end": max(0, start_idx + len(best_sent)),
        }

    return _not_found("I could not find a confident answer to this question in the uploaded paper.")


def _not_found(message: str) -> dict:
    return {
        "answer": message,
        "score": 0.0,
        "found_in_paper": False,
        "start": 0,
        "end": 0,
    }
