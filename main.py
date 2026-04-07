#!/usr/bin/env python3
"""
main.py – Entry point for the Multimodal Phishing Detection Research Prototype.

Workflow:
    1. Load an email (from CLI argument, file, or built-in example).
    2. Start the webcam stream (or mock stream).
    3. Run gaze and face modules on every frame (async loop).
    4. Run the NLP module once on the email text.
    5. Periodically (every FUSION_INTERVAL_S seconds) run the fusion agent
       and print the risk verdict to the console.
    6. Stop on 'q' key-press or after a timeout.

Usage:
    python main.py                           # built-in test email, mock mode
    python main.py --email emails/test.txt   # custom email file
    python main.py --mock                    # force mock mode (no models)
    python main.py --timeout 30              # auto-stop after 30 s
"""

from __future__ import annotations

import argparse
import asyncio
import sys
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

from face_module.openface_extractor import create_face_extractor
from face_module.cognitive_classifier import CognitiveClassifier

from nlp_module.deberta_classifier import DeBERTaPhishingClassifier
from nlp_module.keyword_extractor import scan_keywords

from fusion_module.multimodal_fusion_agent import FusionAgent

logger = get_logger(__name__)


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
    p.add_argument("--email", type=str, default=None, help="Path to email text file")
    p.add_argument("--mock", action="store_true", help="Force mock mode (no heavy models)")
    p.add_argument("--timeout", type=int, default=0, help="Auto-stop after N seconds (0 = manual)")
    p.add_argument("--no-webcam", action="store_true", help="Use mock video stream")
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
        logger.info("Mock mode → L2CSNetEstimator (mock)")
        return L2CSNetEstimator()  # runs in mock internally

    if config.USE_L2CSNET:
        return L2CSNetEstimator()

    # MediaPipe fallback
    try:
        est = MediaPipeGazeEstimator()
        logger.info("L2CS-Net weights unavailable → MediaPipe fallback")
        return est
    except RuntimeError:
        logger.warning("MediaPipe unavailable → L2CS-Net mock mode")
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


# ══════════════════════════════════════════════
# Main async loop
# ══════════════════════════════════════════════
async def run(args: argparse.Namespace) -> None:
    # ── Apply config overrides ───────────────
    if args.mock:
        config.MOCK_MODE = True
    load_overrides()

    # ── Email NLP (run once) ─────────────────
    email_text = load_email(args.email)

    logger.info("Initialising NLP module …")
    nlp = DeBERTaPhishingClassifier()
    nlp_result = nlp.classify(email_text)
    kw_report = scan_keywords(email_text)

    logger.info("═══ NLP Result ═══")
    logger.info("  Phishing probability : %.4f (%s)", nlp_result.phishing_probability, nlp_result.label)
    logger.info("  Suspicious keywords  : %s", nlp_result.suspicious_keywords_found)
    if nlp_result.top_attention_tokens:
        logger.info("  Top attention tokens : %s", nlp_result.top_attention_tokens[:5])

    # ── Video / gaze / face ──────────────────
    logger.info("Initialising gaze module …")
    gaze_estimator = create_gaze_estimator()
    gaze_mapper = GazeMapper()

    logger.info("Initialising face module …")
    face_extractor = create_face_extractor()
    cog_classifier = CognitiveClassifier()

    fusion_agent = FusionAgent()

    # ── Start video ──────────────────────────
    if args.no_webcam or config.MOCK_MODE:
        stream = MockVideoStream().start()
    else:
        try:
            stream = VideoStream().start()
        except RuntimeError:
            logger.warning("Webcam not available → switching to MockVideoStream")
            stream = MockVideoStream().start()

    logger.info("═══ Starting real-time loop (press 'q' to quit) ═══")

    last_fusion_time = time.time()
    start_time = time.time()
    frame_count = 0

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
                    gaze_mapper.add_sample(
                        gaze_result.screen_x,
                        gaze_result.screen_y,
                        gaze_result.confidence,
                    )
                else:
                    mp_result = gaze_estimator.estimate(frame)
                    if mp_result is not None:
                        gaze_mapper.add_sample(
                            mp_result.screen_x,
                            mp_result.screen_y,
                            mp_result.confidence,
                        )

            # ── Face / cognitive ─────────────
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
                cog_state, cog_conf = cog_classifier.predict()

                # Fusion
                verdict = fusion_agent.fuse(
                    phishing_probability=nlp_result.phishing_probability,
                    suspicious_keywords=nlp_result.suspicious_keywords_found,
                    gaze_regions_seen=gaze_state.regions_seen,
                    fixation_times=gaze_state.region_dwell_times,
                    cognitive_state=cog_state,
                    cognitive_confidence=cog_conf,
                )

                elapsed = now - start_time
                print(f"\n{'─' * 60}")
                print(f"  ⏱ t = {elapsed:.1f}s  │  frames = {frame_count}")
                print(f"  👁 Gaze regions seen : {gaze_state.regions_seen}")
                print(f"  🧠 Cognitive state   : {cog_state} {cog_conf}")
                print(f"  📧 Phishing prob     : {nlp_result.phishing_probability:.4f}")
                print(f"\n{verdict.message}")
                print(f"{'─' * 60}\n")

            # ── Timeout ──────────────────────
            if args.timeout > 0 and (now - start_time) >= args.timeout:
                logger.info("Timeout reached (%ds) → stopping", args.timeout)
                break

            # ── Check for quit key ───────────
            # OpenCV key check only works if a window is open; skip in mock
            if not isinstance(stream, MockVideoStream):
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("User pressed 'q' → stopping")
                    break

            await asyncio.sleep(1.0 / config.FPS_TARGET)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt → stopping")
    finally:
        stream.stop()
        if hasattr(face_extractor, "close"):
            face_extractor.close()
        if hasattr(gaze_estimator, "close"):
            gaze_estimator.close()
        cv2.destroyAllWindows()
        logger.info("Shutdown complete.")


# ══════════════════════════════════════════════
# Entry
# ══════════════════════════════════════════════
def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
