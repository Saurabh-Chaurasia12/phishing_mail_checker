from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import google.generativeai as genai
import mss
import numpy as np
import pytesseract
from pytesseract import Output


PROMPT_TEMPLATE = '''You are an expert cybersecurity analyst specializing in phishing detection.

Analyze the email and perform:

1. Classify as "phishing" or "safe"
2. Provide risk score (0 to 1)
3. Extract exact phishing-related phrases

For each phrase include:

* text
* category (urgency, credential_request, financial, suspicious_link, impersonation, threat, other)
* start index
* end index

Return STRICT JSON:

{{
"label": "...",
"risk_score": ...,
"risky_spans": [
{{
"text": "...",
"category": "...",
"start": int,
"end": int
}}
]
}}

Email:
"""
{email_text}
"""
'''

ALLOWED_LABELS = {"phishing", "safe"}
ALLOWED_CATEGORIES = {
    "urgency",
    "credential_request",
    "financial",
    "suspicious_link",
    "impersonation",
    "threat",
    "other",
}


@dataclass
class OCRWordBox:
    text: str
    left: int
    top: int
    width: int
    height: int
    confidence: float  # 0.0 .. 1.0
    block_num: int
    par_num: int
    line_num: int
    word_num: int


@dataclass
class OCRBundle:
    text: str
    words: List[OCRWordBox]
    region_offset: Tuple[int, int]
    crop_region: Optional[Tuple[int, int, int, int]]


def load_dotenv(dotenv_path: Optional[Path] = None) -> None:
    candidate_paths = []
    if dotenv_path is not None:
        candidate_paths.append(dotenv_path)
    else:
        candidate_paths.extend(
            [
                Path.cwd() / ".env",
                Path(__file__).resolve().parent / ".env",
            ]
        )

    for path in candidate_paths:
        if not path.exists():
            continue

        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        return


def resolve_tesseract_path(explicit_path: Optional[str] = None) -> None:
    if explicit_path:
        pytesseract.pytesseract.tesseract_cmd = explicit_path
        return

    discovered = shutil.which("tesseract")
    if discovered:
        pytesseract.pytesseract.tesseract_cmd = discovered
        return

    if os.name == "nt":
        common_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        for path in common_paths:
            if Path(path).exists():
                pytesseract.pytesseract.tesseract_cmd = path
                return


def validate_tesseract() -> None:
    try:
        pytesseract.get_tesseract_version()
    except Exception as exc:
        raise RuntimeError(
            "Tesseract is not available. Install Tesseract OCR and ensure it is on PATH."
        ) from exc


def capture_screenshot(delay_seconds: float = 5.0, monitor_index: int = 1) -> np.ndarray:
    if delay_seconds > 0:
        print(f"Capturing screenshot in {delay_seconds:.1f} seconds...")
        time.sleep(delay_seconds)

    with mss.mss() as sct:
        monitors = sct.monitors
        if monitor_index < 0 or monitor_index >= len(monitors):
            raise ValueError(
                f"Invalid monitor index {monitor_index}. Available range: 0 to {len(monitors) - 1}"
            )
        screenshot = sct.grab(monitors[monitor_index])
        frame = np.array(screenshot, dtype=np.uint8)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)


def preprocess_for_ocr(image: np.ndarray, apply_threshold: bool = True) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if not apply_threshold:
        return gray
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def crop_image_with_offset(
    image: np.ndarray,
    crop_region: Optional[Tuple[int, int, int, int]] = None,
    base_offset: Tuple[int, int] = (0, 0),
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Crop an image and return the translated coordinate offset."""
    if crop_region is None:
        return image, base_offset

    left, top, width, height = crop_region
    if width <= 0 or height <= 0:
        raise ValueError("Crop region width/height must be positive")

    img_h, img_w = image.shape[:2]
    x1 = max(0, min(left, img_w))
    y1 = max(0, min(top, img_h))
    x2 = max(x1, min(left + width, img_w))
    y2 = max(y1, min(top + height, img_h))
    cropped = image[y1:y2, x1:x2]
    return cropped, (base_offset[0] + x1, base_offset[1] + y1)


def _build_text_from_words(words: Sequence[OCRWordBox]) -> str:
    if not words:
        return ""

    ordered = sorted(
        words,
        key=lambda item: (item.block_num, item.par_num, item.line_num, item.word_num),
    )

    lines: List[str] = []
    current_key: Optional[Tuple[int, int, int]] = None
    current_words: List[str] = []
    for word in ordered:
        line_key = (word.block_num, word.par_num, word.line_num)
        if current_key is None:
            current_key = line_key
        if line_key != current_key:
            lines.append(" ".join(current_words))
            current_words = []
            current_key = line_key
        current_words.append(word.text)
    if current_words:
        lines.append(" ".join(current_words))

    return "\n".join(lines).strip()


def extract_email_ocr_bundle_from_image(
    image: np.ndarray,
    min_confidence: float = 30.0,
    apply_threshold: bool = True,
    psm: int = 6,
    region_offset: Tuple[int, int] = (0, 0),
    crop_region: Optional[Tuple[int, int, int, int]] = None,
) -> OCRBundle:
    """Run OCR and return sentence text plus word-level bounding boxes."""
    working_img, translated_offset = crop_image_with_offset(
        image,
        crop_region=crop_region,
        base_offset=region_offset,
    )

    processed = preprocess_for_ocr(working_img, apply_threshold=apply_threshold)
    data = pytesseract.image_to_data(
        processed,
        output_type=Output.DICT,
        config=f"--oem 3 --psm {psm}",
    )

    words: List[OCRWordBox] = []
    for index, raw_text in enumerate(data.get("text", [])):
        text = (raw_text or "").strip()
        if not text:
            continue

        try:
            confidence_percent = float(data["conf"][index])
        except (TypeError, ValueError):
            continue
        if confidence_percent < min_confidence:
            continue

        left = int(data["left"][index]) + translated_offset[0]
        top = int(data["top"][index]) + translated_offset[1]
        width = int(data["width"][index])
        height = int(data["height"][index])

        words.append(
            OCRWordBox(
                text=text,
                left=left,
                top=top,
                width=width,
                height=height,
                confidence=max(0.0, min(1.0, confidence_percent / 100.0)),
                block_num=int(data["block_num"][index]),
                par_num=int(data["par_num"][index]),
                line_num=int(data["line_num"][index]),
                word_num=int(data["word_num"][index]),
            )
        )

    text = _build_text_from_words(words)
    return OCRBundle(
        text=text,
        words=words,
        region_offset=translated_offset,
        crop_region=crop_region,
    )


def save_word_boxes_to_csv(words: Sequence[OCRWordBox], csv_path: str) -> None:
    """Save word-level OCR boxes to CSV for downstream analysis."""
    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "text",
            "left",
            "top",
            "width",
            "height",
            "confidence",
            "block_num",
            "par_num",
            "line_num",
            "word_num",
        ])
        for w in words:
            writer.writerow([
                w.text,
                w.left,
                w.top,
                w.width,
                w.height,
                round(w.confidence, 4),
                w.block_num,
                w.par_num,
                w.line_num,
                w.word_num,
            ])


def extract_email_text_from_image(
    image: np.ndarray,
    min_confidence: float = 30.0,
    apply_threshold: bool = True,
    psm: int = 6,
) -> str:
    bundle = extract_email_ocr_bundle_from_image(
        image,
        min_confidence=min_confidence,
        apply_threshold=apply_threshold,
        psm=psm,
    )
    return bundle.text


def clean_json_response(response_text: str) -> Dict[str, Any]:
    if not response_text or not response_text.strip():
        raise ValueError("Empty response from Gemini.")

    cleaned = response_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in Gemini response.")

    cleaned = cleaned[start : end + 1]
    cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"')
    cleaned = cleaned.replace("\u2018", "'").replace("\u2019", "'")

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON from Gemini: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("Gemini response JSON must be an object.")

    return parsed


def normalize_analysis(result: Dict[str, Any], email_text: str) -> Dict[str, Any]:
    label = str(result.get("label", "")).strip().lower()
    if label not in ALLOWED_LABELS:
        raise ValueError(f"Invalid label in Gemini response: {label!r}")

    risk_score = float(result.get("risk_score", 0.0))
    risk_score = max(0.0, min(1.0, risk_score))

    risky_spans_raw = result.get("risky_spans", [])
    if not isinstance(risky_spans_raw, list):
        raise ValueError("risky_spans must be a list.")

    risky_spans: List[Dict[str, Any]] = []
    email_length = len(email_text)

    for span in risky_spans_raw:
        if not isinstance(span, dict):
            continue

        text = str(span.get("text", ""))
        category = str(span.get("category", "other")).strip().lower()
        if category not in ALLOWED_CATEGORIES:
            category = "other"

        start = int(span.get("start", -1))
        end = int(span.get("end", -1))

        if start < 0 or end < 0 or end <= start or start > email_length or end > email_length:
            match_start = email_text.find(text) if text else -1
            if match_start != -1:
                start = match_start
                end = match_start + len(text)
            else:
                continue

        exact_text = email_text[start:end]
        if text and exact_text != text:
            match_start = email_text.find(text)
            if match_start != -1:
                start = match_start
                end = match_start + len(text)
                exact_text = email_text[start:end]
            else:
                text = exact_text
        else:
            text = exact_text

        risky_spans.append(
            {
                "text": text,
                "category": category,
                "start": start,
                "end": end,
            }
        )

    return {
        "label": label,
        "risk_score": risk_score,
        "risky_spans": risky_spans,
    }


def analyze_email(email_text: str) -> Dict[str, Any]:
    if not email_text or not email_text.strip():
        raise ValueError("Email text is empty.")

    load_dotenv()
    api_key = "api_key"
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable is not set. Add it to the environment or a .env file."
        )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = PROMPT_TEMPLATE.format(email_text=email_text)

    last_error: Optional[Exception] = None
    for attempt in range(3):
        try:
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0,
                    "response_mime_type": "application/json",
                },
            )
            response_text = getattr(response, "text", None)
            if not response_text:
                raise ValueError("Gemini returned an empty response.")
            parsed = clean_json_response(response_text)
            return normalize_analysis(parsed, email_text)
        except Exception as exc:
            last_error = exc
            if attempt == 2:
                break
            time.sleep(1.0)

    raise RuntimeError(f"Failed to analyze email after retries: {last_error}") from last_error


def analyze_screenshot_email(
    delay_seconds: float = 5.0,
    monitor_index: int = 1,
    min_confidence: float = 30.0,
    apply_threshold: bool = True,
    psm: int = 6,
    tesseract_cmd: Optional[str] = None,
) -> Dict[str, Any]:
    resolve_tesseract_path(tesseract_cmd)
    validate_tesseract()
    image = capture_screenshot(delay_seconds=delay_seconds, monitor_index=monitor_index)
    email_text = extract_email_text_from_image(
        image,
        min_confidence=min_confidence,
        apply_threshold=apply_threshold,
        psm=psm,
    )
    if not email_text:
        raise ValueError("No email text detected in the screenshot.")

    analysis = analyze_email(email_text)
    return {
        "email_text": email_text,
        "analysis": analysis,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture a screenshot, OCR email text, and detect phishing using Gemini."
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=5.0,
        help="Seconds to wait before taking the screenshot.",
    )
    parser.add_argument(
        "--monitor",
        type=int,
        default=1,
        help="Monitor index for mss capture. Use 0 for the full virtual desktop, 1+ for a specific monitor.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=30.0,
        help="Minimum OCR confidence for accepted words.",
    )
    parser.add_argument(
        "--psm",
        type=int,
        default=6,
        help="Tesseract page segmentation mode.",
    )
    parser.add_argument(
        "--no-threshold",
        action="store_true",
        help="Disable thresholding and use grayscale-only OCR preprocessing.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run the sample phishing-email test instead of screenshot OCR.",
    )
    parser.add_argument(
        "--tesseract-cmd",
        type=str,
        default=None,
        help="Optional full path to tesseract executable.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.sample:
        sample_email = "Your account has been suspended. Click here to verify your account immediately."
        result = analyze_email(sample_email)
    else:
        result = analyze_screenshot_email(
            delay_seconds=args.delay,
            monitor_index=args.monitor,
            min_confidence=args.min_confidence,
            apply_threshold=not args.no_threshold,
            psm=args.psm,
            tesseract_cmd=args.tesseract_cmd,
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
