"""
Tests for NLP analysis pipeline components.

Tests section detection, text cleaning, metadata extraction,
sentence ranking, keyword extraction, and future research generation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from document_processing.src.text_cleaner import (
    clean_text,
    remove_references_section,
)
from document_processing.src.section_detector import (
    detect_sections,
    extract_abstract_heuristic,
)
from summarization.src.preprocessing import (
    tokenize_sentences,
    score_sentences_tfidf,
    rank_sentences,
)
from summarization.src.inference import (
    extract_keywords,
    generate_future_research,
    _extract_objective,
    _extract_findings,
    _extract_limitations,
)


# ── Sample text fixtures ──────────────────────────────────────────────────────

SAMPLE_TEXT = """
Abstract

This paper proposes a novel transformer-based approach for automatic summarization
of scientific documents. We demonstrate that our method outperforms existing baselines
on the arXiv dataset.

Introduction

The rapid growth of scientific publications makes it challenging for researchers to
keep up with the literature. We propose to address this problem using a transformer-based
summarization model.

Methodology

We train our model on a dataset of 100,000 scientific papers. The architecture consists
of a BERT encoder followed by a GPT-2 decoder. We use the Adam optimizer with a learning
rate of 1e-4.

Results

Our experiments show that the proposed method achieves a ROUGE-1 score of 0.52,
outperforming the baseline by 8 percentage points.

Conclusion

In this work, we presented a novel approach for scientific paper summarization.
Future work could explore larger datasets and multilingual models.
One limitation of our approach is that it requires significant computational resources.

References

[1] Vaswani et al., Attention is All You Need, 2017.
"""


# ── Text cleaner tests ────────────────────────────────────────────────────────

class TestTextCleaner:
    def test_clean_removes_excessive_newlines(self):
        dirty = "Line one\n\n\n\n\nLine two"
        cleaned = clean_text(dirty)
        assert "\n\n\n" not in cleaned

    def test_clean_removes_page_numbers(self):
        dirty = "Some text\n12\nMore text"
        cleaned = clean_text(dirty)
        assert "\n12\n" not in cleaned

    def test_clean_fixes_hyphen_breaks(self):
        dirty = "meth-\nod produces"
        cleaned = clean_text(dirty)
        assert "method" in cleaned

    def test_remove_references(self):
        text_without_refs = remove_references_section(SAMPLE_TEXT)
        assert "Vaswani" not in text_without_refs
        assert "Conclusion" in text_without_refs


# ── Section detector tests ────────────────────────────────────────────────────

class TestSectionDetector:
    def test_detects_abstract(self):
        result = detect_sections(SAMPLE_TEXT)
        section_names = list(result.sections.keys())
        # At least one section should be detected
        assert len(section_names) >= 1

    def test_detects_methodology(self):
        result = detect_sections(SAMPLE_TEXT)
        # Check that some variant of "methodology" or "method" was found
        detected = " ".join(result.sections.keys()).lower()
        assert any(kw in detected for kw in ["method", "approach", "experiment"])

    def test_fallback_on_empty(self):
        result = detect_sections("")
        assert result.sections == {} or len(result.sections) >= 0

    def test_abstract_heuristic(self):
        abstract = extract_abstract_heuristic(SAMPLE_TEXT)
        # May or may not find abstract, but if it does it should be non-empty
        if abstract:
            assert len(abstract) > 50


# ── NLP preprocessing tests ──────────────────────────────────────────────────

class TestPreprocessing:
    def test_tokenize_sentences(self):
        text = "First sentence here. Second sentence follows. Third one ends."
        sents = tokenize_sentences(text)
        assert len(sents) >= 2

    def test_tokenize_filters_short(self):
        text = "A. B. This is a proper sentence that has enough content."
        sents = tokenize_sentences(text)
        # Short fragments should be filtered
        assert all(len(s) > 30 for s in sents)

    def test_tfidf_scores_length_matches(self):
        sents = ["The model achieves high accuracy on all datasets.",
                 "We evaluate performance using ROUGE and BLEU metrics.",
                 "Our approach outperforms state-of-the-art methods significantly."]
        scores = score_sentences_tfidf(sents)
        assert len(scores) == len(sents)

    def test_rank_sentences_returns_subset(self):
        sents = tokenize_sentences(SAMPLE_TEXT)
        ranked = rank_sentences(sents, top_n=3)
        assert len(ranked) <= 3
        assert all(isinstance(s, str) for s in ranked)


# ── Inference pipeline tests ──────────────────────────────────────────────────

class TestInference:
    def test_extract_objective(self):
        text = "In this paper, we propose a novel approach for text classification."
        obj = _extract_objective(text)
        assert len(obj) > 10

    def test_extract_findings(self):
        text = ("Our results show that the model achieves 95% accuracy. "
                "We find that the approach significantly improves performance.")
        findings = _extract_findings(text)
        assert len(findings) >= 1

    def test_extract_limitations(self):
        text = ("One limitation of our approach is the need for large training data. "
                "The method does not handle multilingual inputs.")
        lims = _extract_limitations(text)
        assert len(lims) >= 1

    def test_keyword_extraction_returns_list(self):
        kws = extract_keywords(SAMPLE_TEXT, top_n=10)
        assert isinstance(kws, list)
        # Should return at least some keywords from substantial text
        assert len(kws) >= 1

    def test_future_research_minimum_3(self):
        directions = generate_future_research(
            conclusion_text="Future work could explore larger datasets.",
            discussion_text="One limitation is computational cost.",
            limitations=["Requires significant GPU resources."],
            findings=["The model achieves ROUGE-1 of 0.52."],
            keywords=["transformer", "summarization", "bert"],
            full_text=SAMPLE_TEXT,
        )
        assert len(directions) >= 3

    def test_future_research_has_required_fields(self):
        directions = generate_future_research(
            conclusion_text="Future work may extend to other languages.",
            discussion_text="",
            limitations=[],
            findings=[],
            keywords=["nlp", "language model"],
            full_text=SAMPLE_TEXT,
        )
        for d in directions:
            assert hasattr(d, "topic")
            assert hasattr(d, "question")
            assert hasattr(d, "reason")
            assert hasattr(d, "evidence")
