"""
Core summarization and analysis inference pipeline.

Provides:
1. Extractive summarization (TF-IDF sentence ranking)
2. Abstractive summarization (DistilBART transformer)
3. Structured paper analysis (objective, methodology, findings, etc.)
4. Key insight extraction
5. Future research direction generation
6. Keyword extraction (YAKE)

Models are loaded lazily and cached as module-level singletons.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from summarization.src.preprocessing import (
    rank_sentences,
    tokenize_sentences,
)

logger = logging.getLogger(__name__)

# ── Model singletons ─────────────────────────────────────────────────────────

_summarizer = None
_SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-12-6"

_keyword_extractor = None


def _get_summarizer():
    """Lazy-load the DistilBART summarization pipeline."""
    global _summarizer
    if _summarizer is None:
        logger.info("Loading summarization model: %s", _SUMMARIZER_MODEL)
        from transformers import pipeline
        _summarizer = pipeline(
            "summarization",
            model=_SUMMARIZER_MODEL,
            tokenizer=_SUMMARIZER_MODEL,
        )
        logger.info("Summarization model loaded.")
    return _summarizer


def _get_keyword_extractor():
    """Lazy-load YAKE keyword extractor."""
    global _keyword_extractor
    if _keyword_extractor is None:
        import yake
        _keyword_extractor = yake.KeywordExtractor(
            lan="en",
            n=3,         # max n-gram size
            dedupLim=0.7,
            top=20,
            features=None,
        )
    return _keyword_extractor


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class FutureResearchDirection:
    topic: str
    question: str
    reason: str
    evidence: str
    is_inferred: bool = False    # True if not explicitly stated in the paper


@dataclass
class PaperAnalysis:
    """Full structured analysis output for a research paper."""

    # Paper info (filled from metadata)
    title: str = ""
    authors: List[str] = field(default_factory=list)
    year: Optional[str] = None
    domain: str = ""

    # Core analysis
    summary: str = ""
    objective: str = ""
    methodology: str = ""
    dataset: str = ""
    findings: List[str] = field(default_factory=list)
    main_points: List[str] = field(default_factory=list)
    contribution: str = ""
    limitations: List[str] = field(default_factory=list)
    conclusion: str = ""

    # Key insight
    key_insight: str = ""
    simple_explanation: str = ""
    why_it_matters: str = ""

    # Keywords
    keywords: List[str] = field(default_factory=list)

    # Future research
    future_research: List[FutureResearchDirection] = field(default_factory=list)


# ── Public API ───────────────────────────────────────────────────────────────

def analyze_paper(
    sections: Dict[str, str],
    full_text: str,
    metadata: Optional[dict] = None,
    use_abstractive: bool = True,
) -> PaperAnalysis:
    """
    Run the full analysis pipeline on a segmented paper.

    Args:
        sections: {section_name: text} from SegmentedDocument.
        full_text: Complete cleaned text (fallback when section missing).
        metadata: PaperMetadata dict-like object.
        use_abstractive: Whether to run DistilBART for summary generation.

    Returns:
        PaperAnalysis with all fields populated.
    """
    analysis = PaperAnalysis()

    # ── Populate from metadata ───────────────────────────────────────────────
    if metadata:
        analysis.title = getattr(metadata, "title", "") or ""
        analysis.authors = getattr(metadata, "authors", []) or []
        analysis.year = getattr(metadata, "year", None)

    # ── Extract per-section content ──────────────────────────────────────────
    abstract_text = _get_section(sections, "abstract", full_text, max_chars=3000)
    intro_text = _get_section(sections, "introduction", full_text, max_chars=3000)
    method_text = _get_section(sections, "methodology", full_text, max_chars=4000)
    results_text = _get_section(sections, "results", full_text, max_chars=3000)
    discussion_text = _get_section(sections, "discussion", full_text, max_chars=3000)
    conclusion_text = _get_section(sections, "conclusion", full_text, max_chars=3000)
    experiments_text = _get_section(sections, "experiments", full_text, max_chars=2000)

    # Combined text for summary (excluding references)
    combined = _combine_sections(
        abstract_text, intro_text, method_text,
        results_text, discussion_text, conclusion_text,
        max_chars=8000,
    )

    # ── Objective ────────────────────────────────────────────────────────────
    analysis.objective = _extract_objective(abstract_text or intro_text or full_text[:2000])

    # ── Methodology ──────────────────────────────────────────────────────────
    analysis.methodology = _extract_methodology(method_text or combined)

    # ── Findings ─────────────────────────────────────────────────────────────
    analysis.findings = _extract_findings(
        results_text or experiments_text or conclusion_text or combined
    )

    # ── Main points ──────────────────────────────────────────────────────────
    analysis.main_points = _extract_main_points(combined, sections)

    # ── Contribution ────────────────────────────────────────────────────────
    analysis.contribution = _extract_contribution(
        abstract_text or intro_text or conclusion_text or combined
    )

    # ── Limitations ──────────────────────────────────────────────────────────
    analysis.limitations = _extract_limitations(
        discussion_text or conclusion_text or combined
    )

    # ── Conclusion ───────────────────────────────────────────────────────────
    if conclusion_text:
        sents = tokenize_sentences(conclusion_text)
        analysis.conclusion = " ".join(rank_sentences(sents, section_name="conclusion", top_n=3))
    elif abstract_text:
        sents = tokenize_sentences(abstract_text)
        analysis.conclusion = " ".join(rank_sentences(sents, top_n=2))

    # ── Summary ──────────────────────────────────────────────────────────────
    analysis.summary = _generate_summary(combined, use_abstractive=use_abstractive)

    # ── Dataset ──────────────────────────────────────────────────────────────
    analysis.dataset = _extract_dataset_info(
        method_text or experiments_text or full_text[:5000]
    )

    # ── Domain / keywords ────────────────────────────────────────────────────
    analysis.keywords = extract_keywords(full_text, top_n=15)
    analysis.domain = _infer_domain(analysis.keywords, analysis.title)

    # ── Key insight ──────────────────────────────────────────────────────────
    (
        analysis.key_insight,
        analysis.simple_explanation,
        analysis.why_it_matters,
    ) = _extract_key_insight(
        analysis.contribution,
        analysis.findings,
        analysis.methodology,
        analysis.summary,
    )

    # ── Future research ──────────────────────────────────────────────────────
    analysis.future_research = generate_future_research(
        conclusion_text=conclusion_text,
        discussion_text=discussion_text,
        limitations=analysis.limitations,
        findings=analysis.findings,
        keywords=analysis.keywords,
        full_text=full_text,
    )

    logger.info("Paper analysis complete for '%s'", analysis.title[:60])
    return analysis


# ── Section extraction helpers ───────────────────────────────────────────────

def _get_section(
    sections: Dict[str, str],
    name: str,
    fallback: str,
    max_chars: int = 3000,
) -> str:
    """Return section text, with partial fallbacks for common aliases."""
    # Try primary name and common aliases
    aliases = {
        "methodology": ["methodology", "method", "methods", "approach", "proposed_method"],
        "results": ["results", "findings", "experiments", "evaluation"],
        "conclusion": ["conclusion", "conclusions", "concluding_remarks"],
        "discussion": ["discussion", "analysis"],
        "abstract": ["abstract", "summary"],
        "introduction": ["introduction", "intro"],
    }
    candidates = aliases.get(name, [name])
    for key in candidates:
        if key in sections and sections[key]:
            return sections[key][:max_chars]
        # Partial match (handles "methodology_2" etc.)
        for sec_key in sections:
            if sec_key.startswith(key) and sections[sec_key]:
                return sections[sec_key][:max_chars]
    return ""


def _combine_sections(*texts: str, max_chars: int = 8000) -> str:
    """Concatenate non-empty section texts up to max_chars."""
    combined = "\n\n".join(t for t in texts if t)
    return combined[:max_chars]


# ── Objective extraction ─────────────────────────────────────────────────────

_OBJECTIVE_PATTERNS = [
    r"(?:this paper|this work|this study|we)\s+(?:propose|present|introduce|aim|investigate|address|focus on|explore)\s+([^.]{30,200}\.)",
    r"(?:the|our)\s+(?:goal|objective|purpose|aim)\s+(?:of this (?:paper|work|study)\s+)?(?:is|was)\s+(?:to\s+)?([^.]{30,200}\.)",
    r"in this (?:paper|work|study),?\s+(?:we\s+)?([^.]{30,200}\.)",
]


def _extract_objective(text: str) -> str:
    """Extract the research objective using pattern matching, fallback to sentence ranking."""
    if not text:
        return ""
    for pattern in _OBJECTIVE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # Fallback: top-ranked sentence from abstract/intro
    sents = tokenize_sentences(text)
    ranked = rank_sentences(sents, top_n=2)
    return " ".join(ranked)


# ── Methodology extraction ───────────────────────────────────────────────────

_METHOD_PATTERNS = [
    r"(?:we\s+)?(?:use|employ|adopt|propose|develop|design|implement)\s+([^.]{20,200}\.)",
    r"(?:our\s+)?(?:approach|method|framework|model|architecture|algorithm)\s+(?:is|consists of|involves|uses)\s+([^.]{20,200}\.)",
    r"(?:based on|using|with)\s+([^.]{20,200}\.)(?:\s+we)",
]


def _extract_methodology(text: str) -> str:
    """Extract methodology description."""
    if not text:
        return ""
    for pattern in _METHOD_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).strip()

    sents = tokenize_sentences(text)
    ranked = rank_sentences(sents, section_name="methodology", top_n=3)
    return " ".join(ranked)


# ── Findings extraction ──────────────────────────────────────────────────────

_FINDING_PATTERNS = [
    r"(?:our|the)\s+(?:results?|experiments?|evaluation|findings?|analysis)\s+(?:show|demonstrate|indicate|reveal|suggest)\s+([^.]{20,200}\.)",
    r"(?:we\s+)?(?:find|found|observe|observed|show|showed|demonstrate|demonstrated)\s+(?:that\s+)?([^.]{20,200}\.)",
    r"(?:outperform|achieve|improve|accuracy|f1|precision|recall|bleu|rouge)[^.]{10,200}\.",
]


def _extract_findings(text: str) -> List[str]:
    """Extract a list of key findings from results/experiments text."""
    if not text:
        return []

    findings = []
    for pattern in _FINDING_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            finding = match.group(0).strip()
            if len(finding) > 40 and finding not in findings:
                findings.append(finding)

    if not findings:
        sents = tokenize_sentences(text)
        findings = rank_sentences(sents, section_name="results", top_n=5)

    return findings[:6]


# ── Main points extraction ───────────────────────────────────────────────────

def _extract_main_points(combined_text: str, sections: Dict[str, str]) -> List[str]:
    """Extract the most important sentences across the paper using composite scoring."""
    all_sents: List[str] = []
    # Weight important sections more
    priority_sections = ["abstract", "conclusion", "results", "methodology"]
    for sec in priority_sections:
        for key in sections:
            if key.startswith(sec) and sections[key]:
                sents = tokenize_sentences(sections[key])
                ranked = rank_sentences(sents, section_name=sec, top_n=3)
                all_sents.extend(ranked)

    if not all_sents:
        sents = tokenize_sentences(combined_text)
        all_sents = rank_sentences(sents, top_n=8)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for s in all_sents:
        key = s[:50]
        if key not in seen:
            seen.add(key)
            unique.append(s)

    return unique[:8]


# ── Contribution extraction ──────────────────────────────────────────────────

_CONTRIBUTION_PATTERNS = [
    r"(?:our\s+)?(?:main|key|primary|novel|principal)\s+(?:contribution|contributions?)\s+(?:is|are|include)\s+([^.]{30,300}\.)",
    r"(?:we\s+)?contribute\s+(?:by\s+)?([^.]{30,200}\.)",
    r"(?:to the best of our knowledge|for the first time),?\s+([^.]{20,200}\.)",
    r"(?:novel|new|improved|state.of.the.art)\s+([^.]{20,200}\.)",
]


def _extract_contribution(text: str) -> str:
    """Extract the paper's main contribution statement."""
    if not text:
        return ""
    for pattern in _CONTRIBUTION_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).strip()

    sents = tokenize_sentences(text)
    ranked = rank_sentences(sents, top_n=2)
    return " ".join(ranked)


# ── Limitations extraction ───────────────────────────────────────────────────

_LIMITATION_PATTERNS = [
    r"(?:one\s+)?(?:limitation|drawback|weakness|shortcoming|constraint)\s+(?:of\s+(?:our|this|the)\s+(?:work|approach|method|study)\s+)?(?:is|are)\s+([^.]{20,200}\.)",
    r"(?:our\s+)?(?:approach|method|model)\s+(?:does not|cannot|fails to|is limited)\s+([^.]{20,200}\.)",
    r"(?:future\s+work|future\s+research)\s+(?:should|could|may|might|would)\s+([^.]{20,200}\.)",
    r"(?:we\s+)?(?:did not|do not)\s+(?:address|consider|explore|handle)\s+([^.]{20,200}\.)",
]


def _extract_limitations(text: str) -> List[str]:
    """Extract limitation statements from discussion/conclusion."""
    if not text:
        return []
    limitations = []
    for pattern in _LIMITATION_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            lim = match.group(0).strip()
            if len(lim) > 30 and lim not in limitations:
                limitations.append(lim)
    return limitations[:5]


# ── Dataset extraction ───────────────────────────────────────────────────────

_DATASET_PATTERNS = [
    r"(?:dataset|corpus|benchmark|data)\s+(?:called|named|known as)?\s*([A-Z][A-Za-z0-9\-_]+(?:\s+[A-Z0-9][A-Za-z0-9\-_]+){0,3})",
    r"(?:we\s+)?(?:use|evaluate|train|test)\s+(?:on|with)\s+(?:the\s+)?([A-Z][A-Za-z0-9\-_]+(?:\s+[A-Z0-9][A-Za-z0-9\-_]+){0,3})\s+(?:dataset|benchmark|corpus)",
    r"([A-Z][A-Za-z0-9\-_]+(?:\s+[A-Z0-9][A-Za-z0-9\-_]+){0,3})\s+(?:dataset|benchmark)",
]


def _extract_dataset_info(text: str) -> str:
    """Try to identify datasets mentioned in the paper."""
    if not text:
        return ""
    datasets = []
    for pattern in _DATASET_PATTERNS:
        for match in re.finditer(pattern, text):
            ds = match.group(1).strip()
            if len(ds) > 3 and ds not in datasets:
                datasets.append(ds)
    if datasets:
        return ", ".join(datasets[:5])
    return ""


# ── Summary generation ───────────────────────────────────────────────────────

def _generate_summary(text: str, use_abstractive: bool = True) -> str:
    """
    Generate a summary using extractive + optional abstractive summarization.

    Args:
        text: Combined section text.
        use_abstractive: If True, run DistilBART for final summary.

    Returns:
        Summary string.
    """
    if not text or not text.strip():
        return "Insufficient text to generate summary."

    # Always run extractive first (fast, no model)
    sents = tokenize_sentences(text)
    if not sents:
        return "Insufficient text to generate summary."

    extractive = rank_sentences(sents, top_n=6)
    extractive_summary = " ".join(extractive)

    if not use_abstractive:
        return extractive_summary

    # Try abstractive summarization
    try:
        summarizer = _get_summarizer()
        # DistilBART max input: ~1024 tokens → use first ~3000 chars
        input_text = extractive_summary[:3000] if len(extractive_summary) > 3000 else extractive_summary

        if len(input_text.split()) < 50:
            return extractive_summary

        result = summarizer(
            input_text,
            max_length=200,
            min_length=80,
            do_sample=False,
            truncation=True,
        )
        abstractive = result[0]["summary_text"].strip()
        logger.debug("Abstractive summary generated (%d chars)", len(abstractive))
        return abstractive
    except Exception as exc:
        logger.warning("Abstractive summarization failed, using extractive: %s", exc)
        return extractive_summary


# ── Key insight ──────────────────────────────────────────────────────────────

def _extract_key_insight(
    contribution: str,
    findings: List[str],
    methodology: str,
    summary: str,
) -> tuple[str, str, str]:
    """
    Derive the key insight, simple explanation, and why-it-matters.

    Returns:
        (key_insight, simple_explanation, why_it_matters)
    """
    # Key insight: use contribution if substantial, else top finding
    if contribution and len(contribution) > 50:
        key_insight = contribution
    elif findings:
        key_insight = findings[0]
    elif summary:
        sents = tokenize_sentences(summary)
        key_insight = rank_sentences(sents, top_n=1)[0] if sents else summary[:200]
    else:
        key_insight = ""

    # Simple explanation: strip jargon by using the summary + methodology
    combined_for_simple = (summary + " " + methodology)[:1000]
    sents = tokenize_sentences(combined_for_simple)
    simple_sents = rank_sentences(sents, top_n=2)
    simple_explanation = " ".join(simple_sents) if simple_sents else key_insight

    # Why it matters: drawn from findings
    if len(findings) >= 2:
        why_it_matters = " ".join(findings[:2])
    elif findings:
        why_it_matters = findings[0]
    else:
        why_it_matters = simple_explanation

    return key_insight, simple_explanation, why_it_matters


# ── Keyword extraction ───────────────────────────────────────────────────────

def extract_keywords(text: str, top_n: int = 15) -> List[str]:
    """
    Extract keywords/keyphrases using YAKE.

    Falls back to simple TF-IDF top terms if YAKE fails.
    """
    if not text:
        return []

    try:
        extractor = _get_keyword_extractor()
        keywords = extractor.extract_keywords(text[:8000])
        # YAKE returns (keyphrase, score) — lower score = more relevant
        sorted_kw = sorted(keywords, key=lambda x: x[1])
        return [kw for kw, _ in sorted_kw[:top_n]]
    except Exception as exc:
        logger.warning("YAKE extraction failed: %s — using TF-IDF fallback", exc)
        return _tfidf_keywords(text, top_n)


def _tfidf_keywords(text: str, top_n: int) -> List[str]:
    """Simple TF-IDF keyword extraction fallback."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    try:
        sents = tokenize_sentences(text[:5000])
        if len(sents) < 2:
            return []
        vectorizer = TfidfVectorizer(max_features=200, ngram_range=(1, 2), stop_words="english")
        matrix = vectorizer.fit_transform(sents)
        scores = np.asarray(matrix.sum(axis=0)).flatten()
        vocab = vectorizer.get_feature_names_out()
        top_indices = scores.argsort()[::-1][:top_n]
        return [vocab[i] for i in top_indices]
    except Exception:
        return []


# ── Domain inference ─────────────────────────────────────────────────────────

_DOMAIN_KEYWORDS = {
    "Natural Language Processing": ["nlp", "language model", "text", "bert", "gpt", "transformer", "sentiment", "named entity", "machine translation", "summarization"],
    "Computer Vision": ["image", "object detection", "cnn", "convolutional", "visual", "segmentation", "recognition", "video", "pixel"],
    "Machine Learning": ["machine learning", "neural network", "deep learning", "classification", "regression", "gradient", "overfitting", "training"],
    "Reinforcement Learning": ["reinforcement learning", "reward", "agent", "policy", "q-learning", "environment"],
    "Graph Learning": ["graph neural", "gnn", "knowledge graph", "node", "edge", "graph convolutional"],
    "Bioinformatics": ["protein", "gene", "dna", "rna", "genomic", "biological", "sequence"],
    "Robotics": ["robot", "autonomous", "navigation", "manipulation", "sensor", "actuator"],
    "Data Mining": ["data mining", "pattern", "clustering", "association rule", "anomaly detection"],
}


def _infer_domain(keywords: List[str], title: str) -> str:
    """Infer research domain from keywords and title."""
    combined = " ".join(keywords).lower() + " " + title.lower()
    best_domain = "Computer Science"
    best_score = 0
    for domain, domain_kws in _DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in domain_kws if kw in combined)
        if score > best_score:
            best_score = score
            best_domain = domain
    return best_domain


# ── Future research ──────────────────────────────────────────────────────────

_FUTURE_WORK_PATTERNS = [
    r"(?:future\s+work|future\s+research|in\s+the\s+future)\s+(?:should|could|will|may|might|would|can)\s+([^.]{20,200}\.)",
    r"(?:an?\s+)?(?:interesting|promising|open|important)\s+(?:direction|avenue|challenge|question|problem)\s+(?:for\s+future\s+(?:work|research)\s+)?(?:is|would be)\s+([^.]{20,200}\.)",
    r"(?:remain|remains)\s+(?:an?\s+)?(?:open\s+)?(?:challenge|question|problem|issue)\s+([^.]{10,200}\.)",
    r"(?:we\s+)?(?:did not|do not)\s+(?:address|consider|explore)\s+([^.]{20,150}[,;])",
    r"(?:one\s+limitation|a\s+limitation)\s+(?:of\s+(?:this|our)\s+(?:work|approach|study)\s+)?(?:is\s+that\s+)?([^.]{20,200}\.)",
]


def generate_future_research(
    conclusion_text: str,
    discussion_text: str,
    limitations: List[str],
    findings: List[str],
    keywords: List[str],
    full_text: str,
) -> List[FutureResearchDirection]:
    """
    Generate at least 3 meaningful future research directions.

    Strategy:
    1. Extract explicit future-work statements from conclusion/discussion
    2. Convert limitations into research directions
    3. Generate inferred directions from keywords if needed
    """
    directions: List[FutureResearchDirection] = []
    search_text = (conclusion_text or "") + "\n" + (discussion_text or "")

    # ── Pass 1: Explicit future-work statements ──────────────────────────────
    for pattern in _FUTURE_WORK_PATTERNS:
        for match in re.finditer(pattern, search_text, re.IGNORECASE):
            evidence = match.group(0).strip()
            if len(evidence) < 30:
                continue

            direction = FutureResearchDirection(
                topic=_derive_topic_from_evidence(evidence, keywords),
                question=_derive_question_from_evidence(evidence),
                reason="Explicitly mentioned in the paper as future work.",
                evidence=evidence[:300],
                is_inferred=False,
            )
            if not _is_duplicate(direction, directions):
                directions.append(direction)

            if len(directions) >= 5:
                break

    # ── Pass 2: From limitations ─────────────────────────────────────────────
    for lim in limitations:
        if len(directions) >= 5:
            break
        if len(lim) < 30:
            continue
        direction = FutureResearchDirection(
            topic=f"Addressing: {lim[:60].rstrip('.,')}",
            question=f"How can the limitation — '{lim[:120].rstrip('.,')}'— be overcome?",
            reason="This limitation was identified in the paper and represents a gap to fill.",
            evidence=lim[:300],
            is_inferred=False,
        )
        if not _is_duplicate(direction, directions):
            directions.append(direction)

    # ── Pass 3: Inferred from keywords / findings (if still < 3) ────────────
    if len(directions) < 3:
        inferred = _generate_inferred_directions(keywords, findings, existing=directions)
        directions.extend(inferred)

    return directions[:6]   # cap at 6


def _derive_topic_from_evidence(evidence: str, keywords: List[str]) -> str:
    """Create a short topic label from an evidence sentence."""
    # Use the first noun phrase or the first 8 words
    words = evidence.split()[:8]
    topic = " ".join(words).rstrip(".,;:")
    if keywords:
        for kw in keywords[:5]:
            if kw.lower() in evidence.lower():
                return f"Extending {kw} approaches"
    return topic


def _derive_question_from_evidence(evidence: str) -> str:
    """Transform a future-work statement into a research question."""
    # Replace "we could / should / future work will" with interrogative form
    q = re.sub(
        r"(?:future\s+work|future\s+research)\s+(?:should|could|will|may|might|would|can)\s+",
        "Could researchers ",
        evidence,
        flags=re.IGNORECASE,
    )
    q = re.sub(r"^(?:we|one)\s+(?:could|should|might)\s+", "Could researchers ", q, flags=re.IGNORECASE)
    if not q.endswith("?"):
        q = q.rstrip(".") + "?"
    return q[:250]


def _generate_inferred_directions(
    keywords: List[str],
    findings: List[str],
    existing: List[FutureResearchDirection],
) -> List[FutureResearchDirection]:
    """Generate inferred research directions from keywords when explicit ones are insufficient."""
    directions = []
    needed = 3 - len(existing)
    if needed <= 0:
        return []

    # Template-based inferred directions using top keywords
    templates = [
        {
            "topic": "Scalability and efficiency",
            "question": f"Can the proposed approach scale to larger datasets or more domains ({', '.join(keywords[:2])})?",
            "reason": "Scalability is a common challenge in research; extending the work to larger or more diverse settings is valuable.",
            "evidence": f"Inferred from research domain keywords: {', '.join(keywords[:4])}",
        },
        {
            "topic": "Cross-domain generalization",
            "question": f"How well does the method generalize beyond its tested domain, particularly for {keywords[0] if keywords else 'the studied problem'}?",
            "reason": "Generalization across domains is a key open challenge in most ML research.",
            "evidence": "Inferred from the scope of evaluation described in findings.",
        },
        {
            "topic": "Robustness and fairness evaluation",
            "question": "Are there biases or robustness issues in the proposed approach that could affect real-world deployment?",
            "reason": "Responsible AI research requires evaluation of fairness, bias, and adversarial robustness.",
            "evidence": "Inferred: no explicit fairness/robustness analysis was identified in the paper.",
        },
        {
            "topic": "Real-world application and deployment",
            "question": "What are the engineering and practical challenges in deploying this method in a production system?",
            "reason": "Academic methods often have a gap between prototype and production; exploring this is a natural next step.",
            "evidence": "Inferred from absence of deployment/systems discussion.",
        },
    ]

    for tmpl in templates[:needed]:
        directions.append(
            FutureResearchDirection(
                topic=tmpl["topic"],
                question=tmpl["question"],
                reason=tmpl["reason"],
                evidence=tmpl["evidence"],
                is_inferred=True,
            )
        )

    return directions


def _is_duplicate(candidate: FutureResearchDirection, existing: List[FutureResearchDirection]) -> bool:
    """Simple duplicate check by comparing first 60 chars of evidence."""
    key = candidate.evidence[:60].lower()
    return any(d.evidence[:60].lower() == key for d in existing)
