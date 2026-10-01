"""
NLP preprocessing for the summarization pipeline.

Provides:
- Sentence tokenization
- Stopword removal
- TF-IDF vectorization helpers
- Sentence scoring utilities
"""

from __future__ import annotations

import logging
import re
import string
from typing import List, Optional

import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# ── Stopwords & Tokenization Setup ──────────────────────────────────────────
_DEFAULT_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd",
    "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves"
}

try:
    from nltk.corpus import stopwords as _sw_corpus
    _STOPWORDS = set(_sw_corpus.words("english"))
except Exception:
    _STOPWORDS = _DEFAULT_STOPWORDS

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def tokenize_sentences(text: str) -> List[str]:
    """
    Tokenize text into sentences using NLTK or regex fallback.

    Filters out very short or empty sentences.
    """
    if not text or not text.strip():
        return []
    try:
        sents = nltk.sent_tokenize(text)
    except Exception:
        sents = re.split(r"(?<=[.!?])\s+", text)

    return [s.strip() for s in sents if len(s.strip()) > 10]


def preprocess_sentence(sentence: str) -> str:
    """
    Lowercase, remove punctuation, remove stopwords from a sentence.
    Returns cleaned token string for TF-IDF.
    """
    lower = sentence.lower()
    no_punct = lower.translate(_PUNCT_TABLE)
    tokens = [t for t in no_punct.split() if t not in _STOPWORDS and len(t) > 2]
    return " ".join(tokens)


def compute_tfidf_matrix(sentences: List[str]) -> Optional[np.ndarray]:
    """
    Compute a TF-IDF matrix for a list of raw sentences.

    Returns:
        numpy array (N, vocab_size) or None if fewer than 2 sentences.
    """
    if len(sentences) < 2:
        return None
    cleaned = [preprocess_sentence(s) for s in sentences]
    try:
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(cleaned)
        return matrix.toarray()
    except Exception as exc:
        logger.warning("TF-IDF computation failed: %s", exc)
        return None


def score_sentences_tfidf(sentences: List[str]) -> List[float]:
    """
    Score each sentence by its average TF-IDF weight.

    Returns:
        List of float scores, one per sentence.
    """
    matrix = compute_tfidf_matrix(sentences)
    if matrix is None:
        return [1.0] * len(sentences)
    # Average TF-IDF weight per sentence
    scores = matrix.mean(axis=1).tolist()
    return scores


def score_sentences_similarity(
    sentences: List[str],
    query_sentences: Optional[List[str]] = None,
) -> List[float]:
    """
    Score sentences by cosine similarity to a centroid (or to query sentences).

    If query_sentences is None, uses centroid of all sentences.

    Returns:
        List of float similarity scores.
    """
    matrix = compute_tfidf_matrix(sentences)
    if matrix is None:
        return [1.0] * len(sentences)

    if query_sentences:
        query_cleaned = [preprocess_sentence(s) for s in query_sentences]
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        all_cleaned = [preprocess_sentence(s) for s in sentences] + query_cleaned
        vectorizer.fit(all_cleaned)
        sent_matrix = vectorizer.transform([preprocess_sentence(s) for s in sentences]).toarray()
        query_matrix = vectorizer.transform(query_cleaned).toarray()
        centroid = query_matrix.mean(axis=0, keepdims=True)
    else:
        sent_matrix = matrix
        centroid = matrix.mean(axis=0, keepdims=True)

    similarities = cosine_similarity(sent_matrix, centroid).flatten().tolist()
    return similarities


def score_by_position(sentences: List[str], total_sentences: int) -> List[float]:
    """
    Position-based scoring: sentences at the beginning and end of
    a document/section are more likely to be important.

    Returns:
        Normalized positional weight for each sentence.
    """
    n = len(sentences)
    if n == 0:
        return []

    weights = []
    for i in range(n):
        # Gaussian peak at positions 0 and n-1
        head_score = max(0.0, 1.0 - i / max(n * 0.3, 1))
        tail_score = max(0.0, 1.0 - (n - 1 - i) / max(n * 0.3, 1))
        weights.append(max(head_score, tail_score))

    return weights


def rank_sentences(
    sentences: List[str],
    *,
    section_name: Optional[str] = None,
    top_n: int = 5,
) -> List[str]:
    """
    Rank and return the top-N sentences using a composite score:
      TF-IDF weight × 0.5 + cosine similarity × 0.3 + position × 0.2

    Section importance bonus: abstract/conclusion/results sections are
    given a higher effective weight.

    Args:
        sentences: Raw sentence strings.
        section_name: Canonical section name for bonus scoring.
        top_n: Number of top sentences to return.

    Returns:
        Top-N sentences in their original order.
    """
    if not sentences:
        return []

    n = len(sentences)
    top_n = min(top_n, n)

    # Compute component scores
    tfidf_scores = score_sentences_tfidf(sentences)
    sim_scores = score_sentences_similarity(sentences)
    pos_scores = score_by_position(sentences, n)

    # Normalise each component to [0, 1]
    def _norm(scores):
        arr = np.array(scores)
        span = arr.max() - arr.min()
        if span == 0:
            return np.ones(len(scores))
        return (arr - arr.min()) / span

    tfidf_n = _norm(tfidf_scores)
    sim_n = _norm(sim_scores)
    pos_n = _norm(pos_scores)

    composite = 0.5 * tfidf_n + 0.3 * sim_n + 0.2 * pos_n

    # Section importance bonus
    _IMPORTANT_SECTIONS = {"abstract", "conclusion", "results", "findings"}
    if section_name and section_name.lower() in _IMPORTANT_SECTIONS:
        composite *= 1.2

    # Get top-N indices and preserve original order
    ranked_indices = np.argsort(composite)[::-1][:top_n]
    ranked_indices_sorted = sorted(ranked_indices)

    return [sentences[i] for i in ranked_indices_sorted]
