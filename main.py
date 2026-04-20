#!/usr/bin/env python3
"""
main.py – Entry point for the Multimodal Phishing Detection Research Prototype.

Workflow:
    1. Load an email:
       a) From a text file (--email path/to/file.txt)
       b) From screen capture via OCR (--email-from-screen)
       c) From an image file via OCR (--email-from-image path/to/screenshot.png)
       d) Built-in example (default)
    2. If OCR was used, retain the word bounding boxes for gaze-word tracking.
    3. Start the webcam stream (or mock stream).
    4. Run gaze and face modules on every frame (async loop).
    5. Feed gaze samples to both:
       - GazeMapper  (region-level: sender/subject/body/url/attachments)
       - WordGazeTracker (word-level: which words are being read)
    6. Run the phishing analysis module once on the email text.
    7. Periodically (every FUSION_INTERVAL_S seconds) run the fusion agent
       and print the risk verdict to the console.
    8. Stop on 'q' key-press or after a timeout.

Usage:
    python main.py                                    # built-in test email, mock mode
    python main.py --email-from-screen                # capture screen, OCR, classify
    python main.py --email-from-image screenshot.png  # OCR from image file
    python main.py --email emails/test.txt            # custom email file
    python main.py --mock                             # force mock mode (no models)
    python main.py --timeout 30                       # auto-stop after 30 s
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import hashlib
import os
import time
from typing import Optional

import numpy as np

# ── Project-level imports ────────────────────
import config
from utils.config_loader import load_overrides
from utils.logging_utils import get_logger
from utils.video_stream import MockVideoStream, VideoStream

from gaze_module.gaze360_inference import L2CSNetEstimator
from gaze_module.mediapipe_fallback import MediaPipeGazeEstimator
from gaze_module.gaze_mapper import GazeMapper
from gaze_module.calibration import apply_gaze_calibration, calibrate_gaze_mapping, make_gaze_map_canvas

from face_module.openface_extractor import create_face_extractor
from face_module.cognitive_classifier import CognitiveClassifier

from ocr_module.screen_email_ocr import parse_region, OCRWord
from ocr_module.word_gaze_tracker import WordGazeTracker
from email_phishing_detector import (
    analyze_email,
    capture_screenshot,
    extract_email_ocr_bundle_from_image,
    resolve_tesseract_path,
    save_word_boxes_to_csv,
    validate_tesseract,
)

from fusion_module.multimodal_fusion_agent import FusionAgent
from neon_module.session_runner import run_neon_offline_session

logger = get_logger(__name__)


@dataclass
class LLMPhishingResult:
    phishing_probability: float
    label: str
    suspicious_keywords_found: list[str]
    top_attention_tokens: list[str]


def classify_email_with_llm(email_text: str) -> LLMPhishingResult:
    """Run Gemini-based email analysis and map it to fusion-compatible fields."""
    analysis = analyze_email(email_text)
    risk_score = float(analysis.get("risk_score", 0.0))
    raw_label = str(analysis.get("label", "safe")).strip().lower()
    mapped_label = "phishing" if raw_label == "phishing" else "safe"

    spans = analysis.get("risky_spans", [])
    seen: set[str] = set()
    suspicious_keywords: list[str] = []
    if isinstance(spans, list):
        for span in spans:
            if not isinstance(span, dict):
                continue
            text = str(span.get("text", "")).strip()
            if not text:
                continue
            lowered = text.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            suspicious_keywords.append(text)

    return LLMPhishingResult(
        phishing_probability=max(0.0, min(1.0, risk_score)),
        label=mapped_label,
        suspicious_keywords_found=suspicious_keywords,
        top_attention_tokens=suspicious_keywords[:5],
    )


@dataclass
class OCRRunResult:
    text: str
    words: list[OCRWord]


# ══════════════════════════════════════════════
# Example test email
# ══════════════════════════════════════════════
EXAMPLE_EMAIL = """\
From: security-alert@paypa1.com
Subject: URGENT: Your account has been suspended!

Dear Customer,

We have detected suspicious activity on your PayPal account.
Your account has been temporarily suspended for your security.

To verify your identity and restore access, please click the
link below immediately:

    http://paypa1-secure-login.com/verify?token=abc123

If you do not verify within 24 hours, your account will be
permanently closed and all funds will be forfeited.

Thank you for your prompt attention.

Security Team
PayPal Inc.

Attachment: invoice_details.zip
"""


# ══════════════════════════════════════════════
# Argument parsing
# ══════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multimodal Phishing Detection Prototype")
    # Email source (mutually exclusive)
    email_group = p.add_mutually_exclusive_group()
    email_group.add_argument("--email", type=str, default=None, help="Path to email text file")
    email_group.add_argument("--email-from-screen", action="store_true",
                             help="Capture screen and extract email text via OCR")
    email_group.add_argument("--email-from-image", type=str, default=None,
                             help="Extract email text via OCR from an image file")
    # OCR options
    p.add_argument("--screen-region", type=str, default=None,
                   help="OCR region as left,top,width,height (pixels)")
    p.add_argument("--screen-capture-out", type=str, default=None,
                   help="Optional path to save the captured OCR image")
    # Runtime options
    p.add_argument("--mock", action="store_true", help="Force mock mode (no heavy models)")
    p.add_argument("--timeout", type=int, default=0, help="Auto-stop after N seconds (0 = manual)")
    p.add_argument("--capture-delay", type=int, default=5,
                   help="Seconds to wait before capturing screen (default 5)")
    p.add_argument("--no-webcam", action="store_true", help="Use mock video stream")
    p.add_argument("--show-gaze", action="store_true", help="Show live gaze debug overlay window")
    p.add_argument("--print-gaze", action="store_true", help="Print latest gaze X/Y during fusion updates")
    p.add_argument("--show-email-preview", action="store_true", help="Print short preview of loaded email text")
    p.add_argument("--neon-recording-dir", type=str, default=None,
                   help="Offline Neon recording export directory")
    return p.parse_args()


# ══════════════════════════════════════════════
# Initialisation helpers
# ══════════════════════════════════════════════
def load_email(path: Optional[str]) -> str:
    if path is None:
        logger.info("Using built-in example phishing email")
        return EXAMPLE_EMAIL
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    logger.info("Loaded email from %s (%d chars)", path, len(text))
    return text


def create_gaze_estimator() -> L2CSNetEstimator | MediaPipeGazeEstimator:
    """Pick the best available gaze backend."""
    if config.MOCK_MODE:
        logger.info("Mock mode -> L2CSNetEstimator (mock)")
        return L2CSNetEstimator()  # runs in mock internally

    if config.USE_L2CSNET:
        return L2CSNetEstimator()

    # MediaPipe fallback
    try:
        est = MediaPipeGazeEstimator()
        logger.info("L2CS-Net weights unavailable -> MediaPipe fallback")
        return est
    except RuntimeError:
        logger.warning("MediaPipe unavailable -> L2CS-Net mock mode")
        return L2CSNetEstimator()


# ══════════════════════════════════════════════
# Face crop helper (simple Haar cascade)
# ══════════════════════════════════════════════
import cv2

_face_cascade: Optional[cv2.CascadeClassifier] = None


def _detect_face_crop(frame: np.ndarray) -> Optional[np.ndarray]:
    """Return the largest face crop from *frame*, or None."""
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore[attr-defined]
        _face_cascade = cv2.CascadeClassifier(cascade_path)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    # Largest face
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    return frame[y : y + h, x : x + w]


def print_ocr_report(ocr_result: OCRRunResult, max_words: int = 25) -> None:
    """Print OCR output in a readable, debugging-friendly format."""
    print(f"\n{'=' * 60}")
    print("  [OCR] Extracted Email Text")
    print(f"{'=' * 60}")

    text = ocr_result.text.strip()
    if text:
        print(text)
    else:
        print("<no text extracted>")

    print(f"\n{'-' * 60}")
    print(f"  [OCR] Word Boxes (showing up to {max_words}/{len(ocr_result.words)})")
    print(f"{'-' * 60}")

    if not ocr_result.words:
        print("  <no words detected>")
    else:
        for idx, word in enumerate(ocr_result.words[:max_words], start=1):
            print(
                f"  {idx:02d}. '{word.text}'  conf={word.confidence:.2f}  "
                f"bbox=({word.left},{word.top},{word.width},{word.height})"
            )
        if len(ocr_result.words) > max_words:
            print(f"  ... {len(ocr_result.words) - max_words} more words not shown")

    print(f"{'=' * 60}\n")


# ══════════════════════════════════════════════
# Email loading (Level 1: OCR -> phishing analysis integration)
# ══════════════════════════════════════════════
def load_email_with_ocr(args: argparse.Namespace) -> tuple[str, Optional[OCRRunResult]]:
    """Load email text from the appropriate source.

    Returns (email_text, ocr_result_or_None).
    If OCR was used, ocr_result contains word bounding boxes for Level 2.
    """
    ocr_result: Optional[OCRRunResult] = None
    content_region = config.MAIL_CONTENT_REGION
    resolve_tesseract_path(config.TESSERACT_CMD or None)
    validate_tesseract()

    if args.email_from_screen:
        # ── Level 1: Screen capture -> OCR -> text ──
        region = parse_region(args.screen_region)

        # Countdown so user can position the email on screen
        delay = getattr(args, 'capture_delay', 5)
        if delay > 0:
            print(f"\n{'=' * 50}")
            print(f"  SCREEN CAPTURE in {delay} seconds!")
            print(f"  Open the email you want to analyse NOW.")
            print(f"{'=' * 50}")
            for i in range(delay, 0, -1):
                print(f"  Capturing in {i}...", flush=True)
                import time as _time
                _time.sleep(1)
            print(f"  >>> CAPTURING NOW <<<\n")

        logger.info("Capturing screen for OCR (region=%s)", region)
        screen_img = capture_screenshot(delay_seconds=0.0, monitor_index=1)
        region_offset = (0, 0)
        if region is not None:
            left, top, width, height = region
            img_h, img_w = screen_img.shape[:2]
            x1 = max(0, min(left, img_w))
            y1 = max(0, min(top, img_h))
            x2 = max(x1, min(left + width, img_w))
            y2 = max(y1, min(top + height, img_h))
            screen_img = screen_img[y1:y2, x1:x2]
            region_offset = (x1, y1)

        if args.screen_capture_out:
            cv2.imwrite(args.screen_capture_out, screen_img)
            logger.info("Screen capture saved to %s", args.screen_capture_out)

        ocr_bundle = extract_email_ocr_bundle_from_image(
            screen_img,
            min_confidence=config.OCR_CONFIDENCE_THRESHOLD * 100.0,
            apply_threshold=True,
            psm=6,
            region_offset=region_offset,
            crop_region=content_region,
        )

        save_word_boxes_to_csv(ocr_bundle.words, config.OCR_WORD_BOXES_CSV)
        words = [
            OCRWord(
                text=w.text,
                left=w.left,
                top=w.top,
                width=w.width,
                height=w.height,
                confidence=round(w.confidence, 3),
            )
            for w in ocr_bundle.words
        ]
        ocr_result = OCRRunResult(text=ocr_bundle.text, words=words)
        email_text = ocr_result.text
        logger.info("OCR extracted %d chars, %d words", len(email_text), len(ocr_result.words))
        print(f"[MAIL-CHECK] source=screen_ocr  words={len(ocr_result.words)}")
        print(f"[MAIL-CHECK] word_boxes_csv={config.OCR_WORD_BOXES_CSV}")
        print_ocr_report(ocr_result)

    elif args.email_from_image:
        # ── Level 1: Image file -> OCR -> text ──
        logger.info("Running OCR on image: %s", args.email_from_image)
        print(f"[MAIL-CHECK] Running OCR on {args.email_from_image} …")
        image = cv2.imread(args.email_from_image)
        if image is None:
            raise ValueError(f"Failed to read image: {args.email_from_image}")
        ocr_bundle = extract_email_ocr_bundle_from_image(
            image,
            min_confidence=config.OCR_CONFIDENCE_THRESHOLD * 100.0,
            apply_threshold=True,
            psm=6,
            region_offset=(0, 0),
            crop_region=content_region,
        )
        save_word_boxes_to_csv(ocr_bundle.words, config.OCR_WORD_BOXES_CSV)
        words = [
            OCRWord(
                text=w.text,
                left=w.left,
                top=w.top,
                width=w.width,
                height=w.height,
                confidence=round(w.confidence, 3),
            )
            for w in ocr_bundle.words
        ]
        ocr_result = OCRRunResult(text=ocr_bundle.text, words=words)
        email_text = ocr_result.text
        logger.info("OCR extracted %d chars, %d words", len(email_text), len(ocr_result.words))
        print(f"[MAIL-CHECK] source=image_ocr  words={len(ocr_result.words)}")
        print(f"[MAIL-CHECK] word_boxes_csv={config.OCR_WORD_BOXES_CSV}")
        print_ocr_report(ocr_result)

    else:
        # ── Text file or built-in example ──
        email_text = load_email(args.email)
        if args.email:
            print(f"[MAIL-CHECK] source={args.email}")
            print(f"[MAIL-CHECK] chars={len(email_text)}")

    return email_text, ocr_result


# ══════════════════════════════════════════════
# Main async loop
# ══════════════════════════════════════════════
async def run(args: argparse.Namespace) -> None:
    # ── Apply config overrides ───────────────
    if args.mock:
        config.MOCK_MODE = True
    load_overrides()

    # ── Email loading (Level 1: OCR -> phishing analysis) ───
    email_text, ocr_result = load_email_with_ocr(args)
    email_hash = hashlib.sha256(email_text.encode("utf-8")).hexdigest()[:12]
    logger.info("Email fingerprint (sha256-12): %s", email_hash)
    print(f"[MAIL-CHECK] chars={len(email_text)} sha12={email_hash}")

    if args.show_email_preview:
        preview = " ".join(email_text.strip().split())[:220]
        logger.info("Email preview: %s", preview)
        print(f"[MAIL-CHECK] preview={preview}")

    # ── LLM phishing classification (run once) ────────
    logger.info("Initialising LLM phishing module …")
    phish_result = classify_email_with_llm(email_text)

    logger.info("=== PHISH Result ===")
    logger.info("  Phishing probability : %.4f (%s)", phish_result.phishing_probability, phish_result.label)
    logger.info("  Suspicious keywords  : %s", phish_result.suspicious_keywords_found)
    if phish_result.top_attention_tokens:
        logger.info("  Top attention tokens : %s", phish_result.top_attention_tokens[:5])

    print(f"\n{'=' * 60}")
    print(f"  [PHISH] Classification Result")
    print(f"     Phishing probability : {phish_result.phishing_probability:.4f}")
    print(f"     Label                : {phish_result.label.upper()}")
    print(f"     Suspicious keywords  : {phish_result.suspicious_keywords_found}")
    print(f"{'=' * 60}\n")

    # ── Level 2: WordGazeTracker (if OCR provided word boxes) ──
    word_tracker: Optional[WordGazeTracker] = None
    if ocr_result and ocr_result.words:
        word_tracker = WordGazeTracker(ocr_result.words)
        logger.info("WordGazeTracker active: tracking %d words", len(ocr_result.words))
        print(f"[WORD-TRACK] Tracking {len(ocr_result.words)} words from OCR")
    else:
        logger.info("No OCR word data -> word-level gaze tracking disabled")

    if args.neon_recording_dir:
        if not os.path.isdir(args.neon_recording_dir):
            raise FileNotFoundError(
                f"Neon recording directory not found: {args.neon_recording_dir}"
            )
        logger.info("Neon offline mode enabled -> %s", args.neon_recording_dir)
        result = run_neon_offline_session(
            recording_dir=args.neon_recording_dir,
            phish_result=phish_result,
            ocr_words=ocr_result.words if ocr_result and ocr_result.words else None,
        )
        logger.info("Neon offline processing complete: %d fusion windows", len(result.verdicts))
        return

    # ── Video / gaze / face ──────────────────
    logger.info("Initialising gaze module …")
    gaze_estimator = create_gaze_estimator()
    gaze_mapper = GazeMapper()

    # Start the video stream before calibration so each run learns a fresh mapping.
    if args.no_webcam or config.MOCK_MODE:
        stream = MockVideoStream().start()
    else:
        try:
            stream = VideoStream().start()
        except RuntimeError:
            logger.warning("Webcam not available -> switching to MockVideoStream")
            stream = MockVideoStream().start()

    gaze_calibration = None
    if not args.no_webcam and not config.MOCK_MODE and config.CALIBRATE_GAZE_EVERY_RUN:
        gaze_calibration = calibrate_gaze_mapping(stream, gaze_estimator, _detect_face_crop)
    else:
        logger.info(
            "Skipping gaze calibration (no_webcam=%s, mock_mode=%s, calibrate_enabled=%s)",
            args.no_webcam,
            config.MOCK_MODE,
            config.CALIBRATE_GAZE_EVERY_RUN,
        )

    if config.ENABLE_FACE_MODULE:
        logger.info("Initialising face module …")
        face_extractor = create_face_extractor()
        cog_classifier = CognitiveClassifier()
    else:
        logger.info("Face module disabled via config -> skipping face/cognitive analysis")
        face_extractor = None
        cog_classifier = None

    fusion_agent = FusionAgent()

    logger.info("=== Starting real-time loop (press 'q' to quit) ===")

    last_fusion_time = time.time()
    start_time = time.time()
    frame_count = 0
    latest_gaze: Optional[tuple[float, float, float]] = None  # x, y, confidence in screen coords
    latest_gaze_vec: Optional[tuple[float, float, float]] = None

    try:
        while stream.is_running:
            ts, frame = stream.read()
            if frame is None:
                await asyncio.sleep(0.01)
                continue

            frame_count += 1

            # ── Gaze estimation ──────────────
            face_crop = _detect_face_crop(frame)
            if face_crop is not None:
                if isinstance(gaze_estimator, L2CSNetEstimator):
                    gaze_result = gaze_estimator.estimate(face_crop)
                    if gaze_calibration is not None:
                        calibrated_x, calibrated_y = apply_gaze_calibration(
                            gaze_result.gaze_vector,
                            gaze_calibration,
                        )
                    else:
                        calibrated_x, calibrated_y = gaze_result.screen_x, gaze_result.screen_y

                    latest_gaze = (calibrated_x, calibrated_y, gaze_result.confidence)
                    latest_gaze_vec = gaze_result.gaze_vector
                    gaze_mapper.add_sample(
                        calibrated_x,
                        calibrated_y,
                        gaze_result.confidence,
                    )

                    # ── Level 2: Feed gaze to word tracker ──
                    if word_tracker is not None:
                        word_tracker.add_gaze_sample(calibrated_x, calibrated_y)

                else:
                    mp_result = gaze_estimator.estimate(frame)
                    if mp_result is not None:
                        latest_gaze = (mp_result.screen_x, mp_result.screen_y, mp_result.confidence)
                        gaze_mapper.add_sample(
                            mp_result.screen_x,
                            mp_result.screen_y,
                            mp_result.confidence,
                        )

                        # ── Level 2: Feed gaze to word tracker ──
                        if word_tracker is not None:
                            word_tracker.add_gaze_sample(mp_result.screen_x, mp_result.screen_y)

            # ── Optional live overlay ─────────
            if args.show_gaze:
                debug = frame.copy()
                h, w = debug.shape[:2]

                cv2.rectangle(debug, (10, 10), (w - 10, h - 10), (60, 60, 60), 1)
                cv2.putText(
                    debug,
                    "Webcam view",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

                mini_w, mini_h = 180, 100
                mini_x1 = w - mini_w - 15
                mini_y1 = h - mini_h - 15
                cv2.rectangle(debug, (mini_x1, mini_y1), (mini_x1 + mini_w, mini_y1 + mini_h), (255, 255, 255), 1)
                for x in range(mini_x1, mini_x1 + mini_w, max(mini_w // 4, 1)):
                    cv2.line(debug, (x, mini_y1), (x, mini_y1 + mini_h), (40, 40, 40), 1)
                for y in range(mini_y1, mini_y1 + mini_h, max(mini_h // 3, 1)):
                    cv2.line(debug, (mini_x1, y), (mini_x1 + mini_w, y), (40, 40, 40), 1)

                if latest_gaze is not None:
                    gx, gy, gc = latest_gaze
                    fx = int(np.clip((gx / config.SCREEN_WIDTH) * mini_w, 0, mini_w - 1)) + mini_x1
                    fy = int(np.clip((gy / config.SCREEN_HEIGHT) * mini_h, 0, mini_h - 1)) + mini_y1
                    cv2.circle(debug, (fx, fy), 10, (0, 255, 0), -1)
                    cv2.putText(
                        debug,
                        f"gaze x={gx:.1f} y={gy:.1f} c={gc:.2f}",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                else:
                    cv2.putText(
                        debug,
                        "gaze: no face detected",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 140, 255),
                        2,
                    )

                cv2.imshow("Webcam View", debug)
                cv2.imshow("Gaze Map", make_gaze_map_canvas(latest_gaze))

            # ── Face / cognitive ─────────────
            if face_extractor is not None and cog_classifier is not None:
                face_feats = face_extractor.extract_from_frame(frame)
                cog_classifier.update(face_feats)

            # ── Periodic fusion ──────────────
            now = time.time()
            if now - last_fusion_time >= config.FUSION_INTERVAL_S:
                last_fusion_time = now

                # Gaze state
                gaze_mapper.detect_fixations()
                gaze_state = gaze_mapper.get_state()

                # Cognitive state
                if cog_classifier is not None:
                    cog_state, cog_conf = cog_classifier.predict()
                else:
                    cog_state, cog_conf = "neutral", 0.0

                # Word-level gaze summary (Level 2)
                word_summary = None
                if word_tracker is not None:
                    word_summary = word_tracker.get_reading_summary()

                # Fusion
                verdict = fusion_agent.fuse(
                    phishing_probability=phish_result.phishing_probability,
                    suspicious_keywords=phish_result.suspicious_keywords_found,
                    gaze_regions_seen=gaze_state.regions_seen,
                    fixation_times=gaze_state.region_dwell_times,
                    cognitive_state=cog_state,
                    cognitive_confidence=cog_conf,
                    # Level 2 word-level data
                    words_read=word_summary.words_read if word_summary else None,
                    suspicious_words_gazed=word_summary.suspicious_words_gazed if word_summary else None,
                    unread_suspicious_words=word_summary.unread_suspicious_words if word_summary else None,
                    reading_coverage=word_summary.reading_coverage if word_summary else -1.0,
                )

                elapsed = now - start_time
                print(f"\n{'-' * 60}")
                print(f"  [TIME] t = {elapsed:.1f}s  |  frames = {frame_count}")
                print(f"  [GAZE] Regions seen  : {gaze_state.regions_seen}")
                if args.print_gaze and latest_gaze is not None:
                    print(
                        "  [GAZE] Latest (screen): "
                        f"x={latest_gaze[0]:.1f}, y={latest_gaze[1]:.1f}, conf={latest_gaze[2]:.2f}"
                    )
                    if latest_gaze_vec is not None:
                        print(
                            "  [GAZE] Vector: "
                            f"x={latest_gaze_vec[0]:.3f}, y={latest_gaze_vec[1]:.3f}, z={latest_gaze_vec[2]:.3f}"
                        )
                print(f"  [COG]  Cognitive state: {cog_state} {cog_conf}")
                print(f"  [PHISH] Phishing prob : {phish_result.phishing_probability:.4f}")

                # Level 2: Word reading stats
                if word_summary is not None:
                    print(f"  [READ] Words read    : {word_summary.total_words_read}/{word_summary.total_words_in_email}"
                          f" ({word_summary.reading_coverage:.0%} coverage)")
                    if word_summary.suspicious_words_gazed:
                        print(f"  [READ] Noticed suspicious: {word_summary.suspicious_words_gazed}")
                    if word_summary.unread_suspicious_words:
                        print(f"  [WARN] MISSED suspicious : {word_summary.unread_suspicious_words}")
                    # Top dwell words
                    top_dwell = sorted(
                        word_summary.word_dwell_times.items(),
                        key=lambda x: x[1], reverse=True,
                    )[:5]
                    if top_dwell:
                        dwell_str = ", ".join(f"'{w}'={t:.1f}s" for w, t in top_dwell)
                        print(f"  [READ] Top dwell words: {dwell_str}")

                print(f"\n{verdict.message}")
                print(f"{'-' * 60}\n")

            # ── Timeout ──────────────────────
            if args.timeout > 0 and (now - start_time) >= args.timeout:
                logger.info("Timeout reached (%ds) -> stopping", args.timeout)
                break

            # ── Check for quit key ───────────
            # OpenCV key check only works if a window is open; skip in mock
            if not isinstance(stream, MockVideoStream):
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("User pressed 'q' -> stopping")
                    break

            await asyncio.sleep(1.0 / config.FPS_TARGET)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt -> stopping")
    finally:
        stream.stop()
        if face_extractor is not None and hasattr(face_extractor, "close"):
            face_extractor.close()
        if hasattr(gaze_estimator, "close"):
            gaze_estimator.close()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass  # opencv-python-headless lacks GUI support

        # ── Final word-reading report ────────
        if word_tracker is not None:
            final_summary = word_tracker.get_reading_summary()
            print(f"\n{'=' * 60}")
            print(f"  FINAL WORD-READING REPORT")
            print(f"     Total words in email : {final_summary.total_words_in_email}")
            print(f"     Words read by user   : {final_summary.total_words_read}")
            print(f"     Reading coverage     : {final_summary.reading_coverage:.0%}")
            print(f"     Gaze samples matched : {final_summary.total_gaze_samples}")
            if final_summary.suspicious_words_gazed:
                print(f"     [OK] Noticed suspicious : {final_summary.suspicious_words_gazed}")
            if final_summary.unread_suspicious_words:
                print(f"     [!!] MISSED suspicious  : {final_summary.unread_suspicious_words}")
            print(f"{'=' * 60}\n")

        logger.info("Shutdown complete.")


# ══════════════════════════════════════════════
# Entry
# ══════════════════════════════════════════════
def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
