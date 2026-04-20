"""OCR extraction from screen captures and images.

Supported OCR backends:
    - EasyOCR (default): pure Python model download path
    - pytesseract: wrapper over local Tesseract OCR engine

Main entry points:
    • ``extract_email_from_screen()`` – capture screen region -> OCR -> text + word boxes
    • ``extract_email_from_image()``  – run OCR on an existing image file
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config
from utils.logging_utils import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────
@dataclass
class OCRWord:
    """A single word detected by OCR with its screen-space bounding box."""
    text: str
    left: int          # x of top-left corner (screen px)
    top: int           # y of top-left corner (screen px)
    width: int
    height: int
    confidence: float  # 0.0 – 1.0

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def center(self) -> Tuple[int, int]:
        return self.left + self.width // 2, self.top + self.height // 2

    def contains_point(self, x: float, y: float, padding: int = 5) -> bool:
        """Check if screen point (x, y) falls inside this word's bbox."""
        return (
            (self.left - padding) <= x <= (self.right + padding)
            and (self.top - padding) <= y <= (self.bottom + padding)
        )


@dataclass
class OCRExtractionResult:
    """Full result of an OCR extraction run."""
    text: str                                       # full extracted text
    words: List[OCRWord] = field(default_factory=list)
    image_path: Optional[str] = None                # path to captured/source image
    region_offset: Tuple[int, int] = (0, 0)         # (x, y) offset if region capture
    crop_region: Optional[Tuple[int, int, int, int]] = None  # crop applied before OCR


# ──────────────────────────────────────────────
# Lazy EasyOCR reader singleton
# ──────────────────────────────────────────────
_reader = None


def get_ocr_reader():
    """Return a shared EasyOCR Reader instance (created on first call).

    Models are stored under ``config.OCR_MODEL_DIR`` (pretrained_models/easyocr/).
    """
    global _reader
    if _reader is not None:
        return _reader

    try:
        import easyocr  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "easyocr is required for OCR. Install with: pip install easyocr"
        ) from exc

    os.makedirs(config.OCR_MODEL_DIR, exist_ok=True)
    logger.info(
        "Initialising EasyOCR (languages=%s, gpu=%s, model_dir=%s) …",
        config.OCR_LANGUAGES,
        config.OCR_GPU,
        config.OCR_MODEL_DIR,
    )
    _reader = easyocr.Reader(
        config.OCR_LANGUAGES,
        gpu=config.OCR_GPU,
        model_storage_directory=config.OCR_MODEL_DIR,
        verbose=False,
    )
    logger.info("EasyOCR reader ready")
    return _reader


def _get_pytesseract_module():
    try:
        import pytesseract  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "pytesseract is required for OCR backend 'tesseract'. Install with: pip install pytesseract"
        ) from exc

    if config.TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD
    return pytesseract


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def parse_region(region_text: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    """Parse region string: 'left,top,width,height'."""
    if not region_text:
        return None
    parts = [p.strip() for p in region_text.split(",")]
    if len(parts) != 4:
        raise ValueError("--screen-region must be left,top,width,height")
    left, top, width, height = [int(p) for p in parts]
    if width <= 0 or height <= 0:
        raise ValueError("--screen-region width/height must be positive")
    return left, top, width, height


def capture_screen_image(
    region: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Capture the full screen or a region as a BGR image.

    Returns (image, (offset_x, offset_y)).
    """
    try:
        from PIL import ImageGrab  # type: ignore
    except Exception as exc:
        raise RuntimeError("Pillow ImageGrab is required for screen capture") from exc

    if region is None:
        img = ImageGrab.grab()
        offset = (0, 0)
    else:
        left, top, width, height = region
        bbox = (left, top, left + width, top + height)
        img = ImageGrab.grab(bbox=bbox)
        offset = (left, top)

    rgb = np.array(img)
    if rgb.ndim == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr, offset


def crop_image_with_offset(
    image: np.ndarray,
    crop_region: Optional[Tuple[int, int, int, int]] = None,
    base_offset: Tuple[int, int] = (0, 0),
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Crop an image and translate coordinates back to the original space.

    The returned offset is added to OCR word coordinates so the final boxes are
    still relative to the original image/screen, not the cropped sub-image.
    """
    if crop_region is None:
        return image, base_offset

    left, top, width, height = crop_region
    if width <= 0 or height <= 0:
        raise ValueError("Crop region width/height must be positive")

    h, w = image.shape[:2]
    x1 = max(0, min(left, w))
    y1 = max(0, min(top, h))
    x2 = max(x1, min(left + width, w))
    y2 = max(y1, min(top + height, h))
    cropped = image[y1:y2, x1:x2]
    return cropped, (base_offset[0] + x1, base_offset[1] + y1)


def _run_easyocr(
    image: np.ndarray,
    offset: Tuple[int, int] = (0, 0),
) -> Tuple[str, List[OCRWord]]:
    """Run EasyOCR on an image and return (full_text, list_of_OCRWord).

    Coordinates in the returned OCRWord objects are in *screen-absolute*
    space (offset already applied).
    """
    reader = get_ocr_reader()
    # EasyOCR works on RGB or BGR; it handles conversion internally.
    results = reader.readtext(image)

    words: List[OCRWord] = []
    lines: List[str] = []

    for bbox, text, conf in results:
        if conf < config.OCR_CONFIDENCE_THRESHOLD:
            continue
        if not text.strip():
            continue

        # bbox is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] (four corners)
        xs = [int(pt[0]) for pt in bbox]
        ys = [int(pt[1]) for pt in bbox]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)

        words.append(
            OCRWord(
                text=text.strip(),
                left=x1 + offset[0],
                top=y1 + offset[1],
                width=x2 - x1,
                height=y2 - y1,
                confidence=round(float(conf), 3),
            )
        )
        lines.append(text.strip())

    full_text = "\n".join(lines)
    return full_text, words


def _run_tesseract(
    image: np.ndarray,
    offset: Tuple[int, int] = (0, 0),
) -> Tuple[str, List[OCRWord]]:
    """Run OCR with pytesseract and return (full_text, OCRWord list)."""
    pytesseract = _get_pytesseract_module()

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    data = pytesseract.image_to_data(
        rgb,
        output_type=pytesseract.Output.DICT,
        lang=config.TESSERACT_LANG,
        config="--oem 3 --psm 6",
    )

    words: List[OCRWord] = []
    lines: List[str] = []

    n = len(data.get("text", []))
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue

        conf_raw = data.get("conf", ["-1"])[i]
        try:
            conf_val = float(conf_raw)
        except (TypeError, ValueError):
            conf_val = -1.0

        # Tesseract confidence range is typically 0..100 (or -1 for invalid).
        conf = max(0.0, min(conf_val / 100.0, 1.0))
        if conf < config.OCR_CONFIDENCE_THRESHOLD:
            continue

        left = int(data["left"][i]) + offset[0]
        top = int(data["top"][i]) + offset[1]
        width = int(data["width"][i])
        height = int(data["height"][i])

        words.append(
            OCRWord(
                text=text,
                left=left,
                top=top,
                width=width,
                height=height,
                confidence=round(conf, 3),
            )
        )
        lines.append(text)

    full_text = "\n".join(lines)
    return full_text, words


def _run_ocr(
    image: np.ndarray,
    offset: Tuple[int, int] = (0, 0),
) -> Tuple[str, List[OCRWord]]:
    backend = config.OCR_BACKEND
    if backend in {"tesseract", "pytesseract"}:
        logger.info("OCR backend: pytesseract")
        return _run_tesseract(image, offset=offset)
    if backend != "easyocr":
        logger.warning("Unknown OCR backend '%s'; falling back to easyocr", backend)
    logger.info("OCR backend: easyocr")
    return _run_easyocr(image, offset=offset)


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────
def extract_email_from_screen(
    region: Optional[Tuple[int, int, int, int]] = None,
    crop_region: Optional[Tuple[int, int, int, int]] = None,
    save_capture_path: Optional[str] = None,
) -> OCRExtractionResult:
    """Capture the screen (or a region), run OCR, and return text + word boxes.

    Parameters
    ----------
    region : tuple, optional
        (left, top, width, height) in screen pixels.  ``None`` = full screen.
    save_capture_path : str, optional
        If given, save the captured image to this path.
    """
    image, offset = capture_screen_image(region=region)
    image, offset = crop_image_with_offset(image, crop_region=crop_region, base_offset=offset)

    if save_capture_path:
        cv2.imwrite(save_capture_path, image)
        logger.info("Screen capture saved to %s", save_capture_path)

    full_text, words = _run_ocr(image, offset=offset)
    logger.info("OCR extracted %d words from screen capture", len(words))

    return OCRExtractionResult(
        text=full_text,
        words=words,
        image_path=save_capture_path,
        region_offset=offset,
        crop_region=crop_region,
    )


def extract_email_from_image(
    image_path: str,
    region_offset: Tuple[int, int] = (0, 0),
    crop_region: Optional[Tuple[int, int, int, int]] = None,
) -> OCRExtractionResult:
    """Run OCR on an existing image file.

    Parameters
    ----------
    image_path : str
        Path to the image file (PNG, JPG, etc.).
    region_offset : tuple
        (x, y) offset to add to detected word positions (use if the image
        represents a cropped region of the screen).
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    image, offset = crop_image_with_offset(image, crop_region=crop_region, base_offset=region_offset)

    full_text, words = _run_ocr(image, offset=offset)
    logger.info("OCR extracted %d words from %s", len(words), image_path)

    return OCRExtractionResult(
        text=full_text,
        words=words,
        image_path=image_path,
        region_offset=offset,
        crop_region=crop_region,
    )
