from __future__ import annotations

import os
from dataclasses import dataclass

from pipeline.phishing_analysis import PhishingAnalysisResult
from pipeline.reading_analysis import ReadingAnalysisResult


@dataclass
class WarningResult:
    message: str
    used_llm: bool


def _build_prompt(
    original_email: str,
    phishing_result: PhishingAnalysisResult,
    reading_result: ReadingAnalysisResult,
) -> str:
    phrase_lines = []
    for status in reading_result.phrase_statuses:
        phrase_lines.append(
            f"- phrase: {status.phrase}\n"
            f"  category: {status.category}\n"
            f"  status: {status.status}\n"
            f"  coverage: {status.coverage:.0%}\n"
            f"  read_words: {status.read_words}\n"
            f"  matched_words: {status.matched_words}"
        )

    return (
        "You are helping a user avoid phishing mistakes.\n"
        "Write a concise warning that tells the user which phishing phrases they likely missed while reading.\n"
        "Keep it practical and direct. Mention the most important unread or partially read risky phrases.\n\n"
        f"Email label: {phishing_result.label}\n"
        f"Phishing probability: {phishing_result.phishing_probability:.3f}\n\n"
        "Original email:\n"
        f"{original_email}\n\n"
        "User-read email reconstruction:\n"
        f"{reading_result.user_read_mail}\n\n"
        "Risky phrase reading status:\n"
        f"{chr(10).join(phrase_lines)}\n"
    )


def _fallback_warning(
    phishing_result: PhishingAnalysisResult,
    reading_result: ReadingAnalysisResult,
) -> str:
    missed = reading_result.missed_phrases
    if not missed:
        return (
            "The risky phrases detected in the email appear to have been read. "
            "You should still verify the sender, links, and any urgent requests before acting."
        )

    important = ", ".join(f"'{item.phrase}'" for item in missed[:3])
    return (
        f"This email looks {phishing_result.label} with phishing probability "
        f"{phishing_result.phishing_probability:.2f}. "
        f"You likely missed or only partially read these risky phrases: {important}. "
        "Please go back and review those parts carefully before clicking links, opening attachments, or sharing credentials."
    )


def generate_warning(
    original_email: str,
    phishing_result: PhishingAnalysisResult,
    reading_result: ReadingAnalysisResult,
) -> WarningResult:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return WarningResult(
            message=_fallback_warning(phishing_result, reading_result),
            used_llm=False,
        )

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(
            _build_prompt(original_email, phishing_result, reading_result),
            generation_config={"temperature": 0.2},
        )
        message = (getattr(response, "text", "") or "").strip()
        if not message:
            raise ValueError("Empty warning from Gemini.")
        return WarningResult(message=message, used_llm=True)
    except Exception:
        return WarningResult(
            message=_fallback_warning(phishing_result, reading_result),
            used_llm=False,
        )
