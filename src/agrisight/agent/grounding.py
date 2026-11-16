"""Hallucination guard: flag numbers in an answer that do not appear in its evidence.

In farm advice the dangerous hallucinations are numbers (doses, concentrations,
intervals). Any number the LLM writes must be traceable to the sources, the
farmer's question, or the weather data.
"""
from __future__ import annotations

import re

CITATION_RE = re.compile(r"\[\d+(?:\s*[,-]\s*\d+)*\]")
LIST_MARKER_RE = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)
NUMBER_RE = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])\d+(?:\.\d+)?")


def _norm(num: str) -> str:
    num = num.replace(",", "")
    if "." in num:
        num = num.rstrip("0").rstrip(".")
    return num


def extract_numbers(text: str) -> set[str]:
    text = CITATION_RE.sub(" ", text)
    text = LIST_MARKER_RE.sub(" ", text)
    return {_norm(m.group()) for m in NUMBER_RE.finditer(text)}


def unsupported_numbers(answer: str, evidence: str) -> list[str]:
    return sorted(extract_numbers(answer) - extract_numbers(evidence), key=lambda s: (len(s), s))
