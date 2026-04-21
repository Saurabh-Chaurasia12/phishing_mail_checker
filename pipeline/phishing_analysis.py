from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from email_phishing_detector import analyze_email


@dataclass
class PhishingPhrase:
    text: str
    category: str
    start: int
    end: int


@dataclass
class PhishingAnalysisResult:
    label: str
    phishing_probability: float
    suspicious_phrases: list[PhishingPhrase]
    raw: dict[str, Any]


def classify_email(email_text: str) -> PhishingAnalysisResult:
    analysis = analyze_email(email_text)
    phrases = [
        PhishingPhrase(
            text=str(span.get("text", "")).strip(),
            category=str(span.get("category", "other")).strip().lower(),
            start=int(span.get("start", -1)),
            end=int(span.get("end", -1)),
        )
        for span in analysis.get("risky_spans", [])
        if str(span.get("text", "")).strip()
    ]
    return PhishingAnalysisResult(
        label=str(analysis.get("label", "safe")).strip().lower(),
        phishing_probability=float(analysis.get("risk_score", 0.0)),
        suspicious_phrases=phrases,
        raw=analysis,
    )

