from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional
import re

import cv2

import config
from email_phishing_detector import (
    capture_screenshot,
    extract_email_ocr_bundle_from_image,
    resolve_tesseract_path,
    save_word_boxes_to_csv,
    validate_tesseract,
)
from ocr_module.screen_email_ocr import OCRWord, parse_region
from utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class EmailCaptureResult:
    text: str
    words: list[OCRWord]
    source: str
    image_path: Optional[str] = None
    raw_ocr_text: str = ""


def _normalize_words(words: list[OCRWord]) -> list[OCRWord]:
    return sorted(words, key=lambda word: (word.top, word.left, word.text.lower()))


def _clean_ocr_text(raw_text: str) -> str:
    lines = [line.strip() for line in raw_text.splitlines()]
    cleaned_lines: list[str] = []

    ui_patterns = [
        r"^\s*inbox\b",
        r"^\s*spam\b",
        r"^\s*trash\b",
        r"^\s*starred\b",
        r"^\s*drafts\b",
        r"^\s*sent\b",
        r"^\s*to me\b",
        r"^\s*tome\b",
        r"^\s*reply\b",
        r"^\s*forward\b",
        r"^\s*archive\b",
        r"^\s*more\b",
        r"^\s*[0-9]{1,2}:[0-9]{2}\s*(am|pm)\b",
        r"\bminutes ago\b",
        r"\bhours ago\b",
        r"\bdays ago\b",
    ]

    for line in lines:
        if not line:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue

        lowered = line.lower()
        if any(re.search(pattern, lowered) for pattern in ui_patterns):
            continue

        line = re.sub(r"\bInbox\b.*$", "", line, flags=re.IGNORECASE).strip()
        line = re.sub(r"\bto me\b.*$", "", line, flags=re.IGNORECASE).strip()

        if not line:
            continue

        # Remove lines that are almost entirely UI glyph noise.
        alpha_num = len(re.findall(r"[A-Za-z0-9]", line))
        if alpha_num <= 1 and len(line) <= 3:
            continue

        cleaned_lines.append(line)

    deduped_lines: list[str] = []
    seen_normalized: list[str] = []
    for line in cleaned_lines:
        if line == "":
            if deduped_lines and deduped_lines[-1] != "":
                deduped_lines.append("")
            continue

        normalized = " ".join(re.findall(r"[a-z0-9]+", line.lower()))
        token_count = len(normalized.split())
        alpha_chars = re.findall(r"[A-Za-z]", line)
        upper_ratio = (
            sum(1 for ch in alpha_chars if ch.isupper()) / len(alpha_chars)
            if alpha_chars
            else 0.0
        )
        if token_count <= 3 and upper_ratio >= 0.8:
            continue
        if token_count <= 3 and any(normalized and normalized in prev for prev in seen_normalized):
            continue
        deduped_lines.append(line)
        if normalized:
            seen_normalized.append(normalized)

    while deduped_lines and deduped_lines[0] == "":
        deduped_lines.pop(0)
    while deduped_lines and deduped_lines[-1] == "":
        deduped_lines.pop()

    return "\n".join(deduped_lines).strip()


def _bundle_to_result(bundle, source: str, image_path: Optional[str]) -> EmailCaptureResult:
    words = [
        OCRWord(
            text=word.text,
            left=word.left,
            top=word.top,
            width=word.width,
            height=word.height,
            confidence=round(word.confidence, 3),
        )
        for word in bundle.words
    ]
    normalized_words = _normalize_words(words)
    save_word_boxes_to_csv(bundle.words, config.OCR_WORD_BOXES_CSV)
    raw_text = bundle.text.strip()
    cleaned_text = _clean_ocr_text(raw_text)
    return EmailCaptureResult(
        text=cleaned_text,
        words=normalized_words,
        source=source,
        image_path=image_path,
        raw_ocr_text=raw_text,
    )


def _ocr_quality_score(bundle) -> tuple[float, float, float]:
    text = bundle.text.strip()
    words = bundle.words
    word_count = len(words)
    char_count = len(text)
    avg_conf = sum(word.confidence for word in words) / max(word_count, 1)
    alpha_chars = len(re.findall(r"[A-Za-z0-9]", text))
    weird_chars = len(re.findall(r"[^A-Za-z0-9\s,.:;!?@/_()'\-]", text))
    garbage_penalty = weird_chars / max(alpha_chars + weird_chars, 1)
    line_count = len([line for line in text.splitlines() if line.strip()])

    # Prefer richer OCR output with decent confidence and fewer malformed symbols.
    score = (
        word_count * 4.0
        + min(char_count, 2500) * 0.03
        + line_count * 1.5
        + avg_conf * 20.0
        - garbage_penalty * 30.0
    )
    return score, avg_conf, garbage_penalty


def _extract_best_ocr_bundle(image, region_offset=(0, 0)):
    candidate_crops: list[tuple[str, Optional[tuple[int, int, int, int]]]] = [("full_image", None)]
    if config.MAIL_CONTENT_REGION:
        candidate_crops.append(("mail_content_region", config.MAIL_CONTENT_REGION))

    candidates = []
    for label, crop_region in candidate_crops:
        bundle = extract_email_ocr_bundle_from_image(
            image,
            min_confidence=config.OCR_CONFIDENCE_THRESHOLD * 100.0,
            apply_threshold=True,
            psm=6,
            region_offset=region_offset,
            crop_region=crop_region,
        )
        score, avg_conf, garbage_penalty = _ocr_quality_score(bundle)
        logger.info(
            "OCR candidate '%s': words=%d chars=%d score=%.2f conf=%.3f garbage=%.3f",
            label,
            len(bundle.words),
            len(bundle.text.strip()),
            score,
            avg_conf,
            garbage_penalty,
        )
        candidates.append((score, label, bundle))

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_label, best_bundle = candidates[0]
    logger.info("Selected OCR candidate '%s' (score=%.2f)", best_label, best_score)
    return best_bundle


def load_email_text(path: str) -> EmailCaptureResult:
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    logger.info("Loaded email text from %s", path)
    return EmailCaptureResult(text=text, words=[], source=path, image_path=None)


def capture_email_from_screen(
    delay_seconds: int = 5,
    screen_region: Optional[str] = None,
    capture_out: Optional[str] = None,
    monitor_index: int = 1,
) -> EmailCaptureResult:
    resolve_tesseract_path(config.TESSERACT_CMD or None)
    validate_tesseract()

    region = parse_region(screen_region)
    if delay_seconds > 0:
        print(f"\nPreparing screen capture. Screenshot will be taken in {delay_seconds} second(s).")
        for remaining in range(delay_seconds, 0, -1):
            print(f"  Capturing in {remaining}...", flush=True)
            time.sleep(1)

    screenshot = capture_screenshot(delay_seconds=0.0, monitor_index=monitor_index)
    region_offset = (0, 0)
    if region is not None:
        left, top, width, height = region
        height_px, width_px = screenshot.shape[:2]
        x1 = max(0, min(left, width_px))
        y1 = max(0, min(top, height_px))
        x2 = max(x1, min(left + width, width_px))
        y2 = max(y1, min(top + height, height_px))
        screenshot = screenshot[y1:y2, x1:x2]
        region_offset = (x1, y1)

    if capture_out:
        cv2.imwrite(capture_out, screenshot)
        logger.info("Saved screenshot to %s", capture_out)

    bundle = _extract_best_ocr_bundle(screenshot, region_offset=region_offset)
    logger.info("OCR extracted %d words from screen capture", len(bundle.words))
    return _bundle_to_result(bundle, source="screen_capture", image_path=capture_out)


def capture_email_from_image(image_path: str) -> EmailCaptureResult:
    resolve_tesseract_path(config.TESSERACT_CMD or None)
    validate_tesseract()

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    bundle = _extract_best_ocr_bundle(image, region_offset=(0, 0))
    logger.info("OCR extracted %d words from %s", len(bundle.words), image_path)
    return _bundle_to_result(bundle, source=image_path, image_path=image_path)
