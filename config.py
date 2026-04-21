"""Configuration for the offline phishing-reading pipeline."""

from __future__ import annotations

import os

try:
    import torch  # type: ignore
except ImportError:
    torch = None


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "pretrained_models")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

OCR_BACKEND = os.environ.get("MTP_OCR_BACKEND", "tesseract").lower()
OCR_MODEL_DIR = os.path.join(MODEL_DIR, "easyocr")
OCR_LANGUAGES = ["en"]
OCR_GPU = bool(torch and torch.cuda.is_available())
OCR_CONFIDENCE_THRESHOLD = float(os.environ.get("MTP_OCR_CONFIDENCE_THRESHOLD", "0.3"))
TESSERACT_CMD = os.environ.get("MTP_TESSERACT_CMD", "")
TESSERACT_LANG = os.environ.get("MTP_TESSERACT_LANG", "eng")
OCR_WORD_BOXES_CSV = os.environ.get(
    "MTP_OCR_WORD_BOXES_CSV",
    os.path.join(LOG_DIR, "ocr_word_boxes.csv"),
)

MAIL_CONTENT_REGION = (190, 190, 1540, 840)
SUSPICIOUS_KEYWORDS = [
    "urgent",
    "verify",
    "click here",
    "suspended",
    "password",
    "account",
    "security",
    "update",
    "confirm",
    "immediately",
    "expire",
    "login",
    "credentials",
    "bank",
    "paypal",
    "attachment",
]

LOG_LEVEL = os.environ.get("MTP_LOG_LEVEL", "INFO")
LOG_TO_FILE = True
