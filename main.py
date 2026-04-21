#!/usr/bin/env python3
"""Offline phishing-reading pipeline entrypoint."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

import config
from pipeline.email_capture import (
    EmailCaptureResult,
    capture_email_from_image,
    capture_email_from_screen,
    load_email_text,
)
from pipeline.gaze_input import load_gaze_samples_from_xlsx
from pipeline.phishing_analysis import classify_email
from pipeline.reading_analysis import analyze_reading
from pipeline.warning_agent import generate_warning
from utils.config_loader import load_overrides
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline OCR + gaze replay phishing warning pipeline")
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--email", type=str, default=None, help="Path to a plain-text email file")
    source_group.add_argument("--email-from-image", type=str, default=None, help="Path to an email screenshot/image")
    source_group.add_argument(
        "--email-from-screen",
        action="store_true",
        help="Capture a screenshot after a countdown and OCR it",
    )
    parser.add_argument("--capture-delay", type=int, default=5, help="Seconds before screenshot capture")
    parser.add_argument("--screen-region", type=str, default=None, help="Screen region as left,top,width,height")
    parser.add_argument(
        "--screen-capture-out",
        type=str,
        default=os.path.join(config.LOG_DIR, "email_screenshot.png"),
        help="Optional path to save the captured screenshot",
    )
    parser.add_argument(
        "--gaze-xlsx",
        type=str,
        default=os.path.join("neon_data", "time.xlsx"),
        help="Excel file with timestamp,x,y gaze samples",
    )
    parser.add_argument(
        "--min-read-ms",
        type=float,
        default=20.0,
        help="Minimum dwell time in milliseconds required to count a word as read",
    )
    parser.add_argument(
        "--report-out",
        type=str,
        default=os.path.join(config.LOG_DIR, "pipeline_report.json"),
        help="Optional JSON report path",
    )
    parser.add_argument(
        "--analysis-log-out",
        type=str,
        default=os.path.join(config.LOG_DIR, "pipeline_analysis.log"),
        help="Human-readable log file for extracted email, reading analysis, and warning output",
    )
    return parser.parse_args()


def _load_email_source(args: argparse.Namespace) -> EmailCaptureResult:
    if args.email:
        return load_email_text(args.email)
    if args.email_from_image:
        return capture_email_from_image(args.email_from_image)
    return capture_email_from_screen(
        delay_seconds=args.capture_delay,
        screen_region=args.screen_region,
        capture_out=args.screen_capture_out if args.email_from_screen or not args.email else None,
    )


def _print_summary(email_result, phishing_result, reading_result, warning_result) -> None:
    print(f"\n[EMAIL] Source: {email_result.source}")
    print(f"[EMAIL] OCR words: {len(email_result.words)}")
    print(f"[PHISH] Label: {phishing_result.label}")
    print(f"[PHISH] Probability: {phishing_result.phishing_probability:.4f}")

    if phishing_result.suspicious_phrases:
        print("[PHISH] Suspicious phrases:")
        for phrase in phishing_result.suspicious_phrases:
            print(f"  - {phrase.text} ({phrase.category})")
    else:
        print("[PHISH] Suspicious phrases: none")

    print(f"[READ] Total gaze samples: {reading_result.total_samples}")
    print(f"[READ] Samples matched to OCR words: {reading_result.matched_samples}")
    print(f"[READ] Words counted as read: {len(reading_result.read_word_indices)}")

    if reading_result.missed_phrases:
        print("[READ] Missed or partial risky phrases:")
        for status in reading_result.missed_phrases:
            print(f"  - {status.phrase} -> {status.status} ({status.coverage:.0%} coverage)")
    else:
        print("[READ] All risky phrases appear to have been read.")

    print("\n[WARNING]")
    print(warning_result.message)
    print(f"\n[WARNING] Generated with {'LLM' if warning_result.used_llm else 'fallback logic'}")


def _format_phrase_block(reading_result) -> str:
    if not reading_result.phrase_statuses:
        return "No suspicious phrases were returned by the phishing model."

    lines: list[str] = []
    for index, status in enumerate(reading_result.phrase_statuses, start=1):
        lines.extend(
            [
                f"{index}. Phrase: {status.phrase}",
                f"   Category: {status.category}",
                f"   Status: {status.status}",
                f"   Coverage: {status.coverage:.0%}",
                f"   Matched OCR words: {status.matched_words}",
                f"   Read words: {status.read_words}",
            ]
        )
    return "\n".join(lines)


def _top_dwell_lines(email_result, reading_result, limit: int = 20) -> str:
    if not reading_result.word_dwell_times:
        return "No word dwell times were recorded."

    ranked = sorted(
        reading_result.word_dwell_times.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:limit]
    lines = []
    for word_index, dwell in ranked:
        word_text = email_result.words[word_index].text if 0 <= word_index < len(email_result.words) else "<unknown>"
        lines.append(f"- {word_text}: {dwell:.3f}s")
    return "\n".join(lines)


def _write_analysis_log(args, email_result, phishing_result, reading_result, warning_result) -> None:
    log_dir = os.path.dirname(args.analysis_log_out)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    content = f"""\
========================================================================
PHISHING MAIL CHECKER ANALYSIS LOG
========================================================================

INPUTS
------------------------------------------------------------------------
Email source       : {email_result.source}
Image path         : {email_result.image_path}
Gaze xlsx          : {args.gaze_xlsx}
Min read threshold : {args.min_read_ms:.1f} ms
OCR word count     : {len(email_result.words)}

EXTRACTED EMAIL TEXT
------------------------------------------------------------------------
{email_result.text.strip() or "<empty>"}

RAW OCR TEXT
------------------------------------------------------------------------
{email_result.raw_ocr_text.strip() or "<empty>"}

READING SUMMARY
------------------------------------------------------------------------
Total gaze samples           : {reading_result.total_samples}
Samples matched to OCR words : {reading_result.matched_samples}
Words with any hit           : {len(reading_result.words_with_hits)}
Words counted as read        : {len(reading_result.read_word_indices)}

USER READ MAIL
------------------------------------------------------------------------
{reading_result.user_read_mail.strip() or "<no words met the read threshold>"}

WORDS WITH HITS
------------------------------------------------------------------------
{reading_result.words_with_hits or "<none>"}

TOP WORD DWELL TIMES
------------------------------------------------------------------------
{_top_dwell_lines(email_result, reading_result)}

PHISHING CLASSIFICATION
------------------------------------------------------------------------
Label               : {phishing_result.label}
Phishing probability: {phishing_result.phishing_probability:.4f}

PHISHING PHRASES
------------------------------------------------------------------------
{_format_phrase_block(reading_result)}

FINAL WARNING
------------------------------------------------------------------------
Generated with : {"LLM" if warning_result.used_llm else "fallback logic"}
{warning_result.message}

JSON REPORT
------------------------------------------------------------------------
{args.report_out}
"""

    with open(args.analysis_log_out, "w", encoding="utf-8") as handle:
        handle.write(content)
    logger.info("Saved human-readable analysis log to %s", args.analysis_log_out)


def main() -> None:
    args = parse_args()
    load_overrides()

    email_result = _load_email_source(args)
    if not email_result.text.strip():
        raise ValueError("No email text was extracted.")

    phishing_result = classify_email(email_result.text)

    gaze_samples = load_gaze_samples_from_xlsx(args.gaze_xlsx)
    reading_result = analyze_reading(
        words=email_result.words,
        gaze_samples=gaze_samples,
        phishing_result=phishing_result,
        min_read_time_s=max(args.min_read_ms / 1000.0, 0.0),
    )
    warning_result = generate_warning(email_result.text, phishing_result, reading_result)

    report_dir = os.path.dirname(args.report_out)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(args.report_out, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "email_source": asdict(email_result),
                "phishing_result": asdict(phishing_result),
                "reading_result": asdict(reading_result),
                "warning_result": asdict(warning_result),
            },
            handle,
            indent=2,
        )
    logger.info("Saved pipeline report to %s", args.report_out)
    _write_analysis_log(args, email_result, phishing_result, reading_result, warning_result)

    _print_summary(email_result, phishing_result, reading_result, warning_result)


if __name__ == "__main__":
    main()
