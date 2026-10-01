"""
Research-paper section detection and segmentation.

Identifies standard sections such as Abstract, Introduction,
Related Work, Methodology, Results, Discussion, Conclusion, etc.

Does NOT assume a fixed structure — uses heuristic regex matching
against common academic section headings.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Section label taxonomy ────────────────────────────────────────────────────
# Maps canonical name → list of regex patterns that match it.
# Patterns are matched against the *cleaned* heading text (case-insensitive).

_SECTION_PATTERNS: List[Tuple[str, List[str]]] = [
    ("abstract", [r"^abstract$", r"^summary$"]),
    ("introduction", [r"^introduction$", r"^1\.\s*introduction", r"^overview$"]),
    ("related_work", [
        r"^related\s+work", r"^literature\s+review",
        r"^background", r"^prior\s+work", r"^previous\s+work",
        r"^state\s+of\s+the\s+art",
    ]),
    ("methodology", [
        r"^method", r"^methodology", r"^approach",
        r"^proposed\s+(method|approach|framework|model)",
        r"^(system\s+)?design", r"^experimental\s+setup",
        r"^materials\s+and\s+methods",
    ]),
    ("experiments", [
        r"^experiment", r"^evaluation", r"^empirical",
        r"^benchmark", r"^setup",
    ]),
    ("results", [
        r"^results?$", r"^findings?$", r"^performance",
        r"^quantitative\s+results",
    ]),
    ("discussion", [
        r"^discussion", r"^analysis", r"^ablation",
        r"^qualitative\s+analysis",
    ]),
    ("conclusion", [
        r"^conclusion", r"^concluding\s+remarks",
        r"^summary\s+and\s+conclusion", r"^future\s+work",
    ]),
    ("references", [r"^references?$", r"^bibliography$", r"^works\s+cited"]),
    ("acknowledgments", [r"^acknowledge?ment", r"^thanks"]),
    ("appendix", [r"^appendix", r"^supplementary"]),
]

# Pre-compile: (canonical_name, compiled_pattern)
_COMPILED: List[Tuple[str, re.Pattern]] = [
    (name, re.compile("|".join(pats), re.IGNORECASE))
    for name, pats in _SECTION_PATTERNS
]

# A heading line is short and starts with a common heading pattern
_HEADING_LINE = re.compile(
    r"^(\d+\.?\d*\.?\s+)?[A-Z][A-Za-z\s\-:&/]{2,60}$"
)


@dataclass
class Section:
    """Represents a detected section of a research paper."""

    name: str               # canonical label e.g. "methodology"
    heading: str            # the original heading text found in the doc
    text: str               # full text of this section
    start_line: int         # line number in the full text where section starts
    end_line: int           # inclusive end line number


@dataclass
class SegmentedDocument:
    """Full segmented document."""

    sections: Dict[str, Section] = field(default_factory=dict)
    unmatched_text: str = ""

    def get(self, section_name: str) -> Optional[str]:
        """Return the text of a section, or None if not found."""
        sec = self.sections.get(section_name)
        return sec.text if sec else None

    def get_multiple(self, *names: str) -> str:
        """
        Return concatenated text of multiple sections (in order given).
        Skips missing sections silently.
        """
        parts = []
        for name in names:
            text = self.get(name)
            if text:
                parts.append(text)
        return "\n\n".join(parts)

    def section_names(self) -> List[str]:
        return list(self.sections.keys())


def _match_heading(line: str) -> Optional[str]:
    """
    Try to match a line against known section headings.
    Returns canonical name or None.
    """
    stripped = line.strip()
    for name, pattern in _COMPILED:
        if pattern.search(stripped):
            return name
    return None


def _is_heading_candidate(line: str) -> bool:
    """Heuristic: is this line likely a section heading?"""
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) > 80:
        return False
    if stripped.endswith(".") and not re.match(r"^\d", stripped):
        return False   # regular sentence ending with period
    return bool(_HEADING_LINE.match(stripped))


def detect_sections(clean_text: str) -> SegmentedDocument:
    """
    Detect and segment sections from cleaned document text.

    Args:
        clean_text: The cleaned full text of a research paper.

    Returns:
        SegmentedDocument with detected sections.
    """
    lines = clean_text.split("\n")
    total_lines = len(lines)

    # Collect (line_idx, canonical_name, original_heading) for each detected heading
    boundaries: List[Tuple[int, str, str]] = []

    for idx, line in enumerate(lines):
        if not _is_heading_candidate(line):
            continue
        canonical = _match_heading(line)
        if canonical:
            boundaries.append((idx, canonical, line.strip()))
            logger.debug("Section detected: '%s' at line %d", canonical, idx)

    if not boundaries:
        logger.warning("No recognizable section headings found in document.")
        return SegmentedDocument(unmatched_text=clean_text)

    # Build sections from boundaries
    sections: Dict[str, Section] = {}
    for i, (start_idx, name, heading) in enumerate(boundaries):
        end_idx = boundaries[i + 1][0] - 1 if i + 1 < len(boundaries) else total_lines - 1
        # Slice lines (skip the heading line itself)
        section_lines = lines[start_idx + 1 : end_idx + 1]
        text = "\n".join(section_lines).strip()

        # If duplicate section name, append suffix to avoid overwriting
        canonical = name
        if canonical in sections:
            canonical = f"{name}_{i}"

        sections[canonical] = Section(
            name=canonical,
            heading=heading,
            text=text,
            start_line=start_idx,
            end_line=end_idx,
        )

    # Unmatched text: lines before the first detected heading
    first_boundary = boundaries[0][0]
    unmatched = "\n".join(lines[:first_boundary]).strip()

    logger.info(
        "Section detection complete: %d sections found: %s",
        len(sections),
        list(sections.keys()),
    )

    return SegmentedDocument(sections=sections, unmatched_text=unmatched)


def extract_abstract_heuristic(text: str) -> Optional[str]:
    """
    Fallback: Try to extract an abstract from the raw text even if
    section detection failed, using the word 'Abstract' as an anchor.

    Returns the abstract text or None.
    """
    match = re.search(r"\bAbstract\b[\s:—\-]*\n?(.*?)(?=\n\n|\n[A-Z][A-Za-z\s]{2,40}\n)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None
