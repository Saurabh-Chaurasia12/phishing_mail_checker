"""
gaze_module/calibration.py - Gaze calibration and visualization helpers.

This module keeps calibration logic separate from main runtime orchestration.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import cv2
import numpy as np

import config
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def build_gaze_calibration_targets() -> list[tuple[float, float]]:
    """Return normalized screen targets for calibration."""
    if config.GAZE_CALIBRATION_POINTS == 9:
        return [
            (0.15, 0.15), (0.50, 0.15), (0.85, 0.15),
            (0.15, 0.50), (0.50, 0.50), (0.85, 0.50),
            (0.15, 0.85), (0.50, 0.85), (0.85, 0.85),
        ]
    return [
        (0.50, 0.50),
        (0.15, 0.15),
        (0.85, 0.15),
        (0.15, 0.85),
        (0.85, 0.85),
    ]


def fit_gaze_calibration(
    samples: list[tuple[float, float, float, float]],
) -> Optional[dict[str, np.ndarray]]:
    """Fit an affine mapping from gaze vector x/y to screen x/y."""
    if len(samples) < config.GAZE_CALIBRATION_MIN_SAMPLES:
        return None

    design = np.array([[gx, gy, 1.0] for gx, gy, _, _ in samples], dtype=np.float32)
    target_x = np.array([sx for _, _, sx, _ in samples], dtype=np.float32)
    target_y = np.array([sy for _, _, _, sy in samples], dtype=np.float32)

    coeff_x, *_ = np.linalg.lstsq(design, target_x, rcond=None)
    coeff_y, *_ = np.linalg.lstsq(design, target_y, rcond=None)
    return {"x": coeff_x, "y": coeff_y}


def apply_gaze_calibration(
    gaze_vector: tuple[float, float, float],
    calibration: Optional[dict[str, np.ndarray]],
) -> tuple[float, float]:
    """Map a gaze vector to calibrated screen coordinates."""
    if calibration is None:
        return 0.0, 0.0

    design = np.array([gaze_vector[0], gaze_vector[1], 1.0], dtype=np.float32)
    screen_x = float(np.clip(design @ calibration["x"], 0, config.SCREEN_WIDTH))
    screen_y = float(np.clip(design @ calibration["y"], 0, config.SCREEN_HEIGHT))
    return screen_x, screen_y


def make_gaze_map_canvas(latest_gaze: Optional[tuple[float, float, float]]) -> np.ndarray:
    """Create a screen-space visualization canvas."""
    display_w = min(config.SCREEN_WIDTH, 1280)
    display_h = min(config.SCREEN_HEIGHT, 720)
    canvas = np.zeros((display_h, display_w, 3), dtype=np.uint8)

    step_x = max(display_w // 8, 1)
    step_y = max(display_h // 6, 1)
    for x in range(0, display_w, step_x):
        cv2.line(canvas, (x, 0), (x, display_h), (40, 40, 40), 1)
    for y in range(0, display_h, step_y):
        cv2.line(canvas, (0, y), (display_w, y), (40, 40, 40), 1)

    cv2.line(canvas, (display_w // 2, 0), (display_w // 2, display_h), (80, 80, 80), 1)
    cv2.line(canvas, (0, display_h // 2), (display_w, display_h // 2), (80, 80, 80), 1)

    cv2.putText(
        canvas,
        f"screen {config.SCREEN_WIDTH} x {config.SCREEN_HEIGHT}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (220, 220, 220),
        2,
    )

    if latest_gaze is not None:
        gx, gy, gc = latest_gaze
        fx = int(np.clip((gx / config.SCREEN_WIDTH) * display_w, 0, display_w - 1))
        fy = int(np.clip((gy / config.SCREEN_HEIGHT) * display_h, 0, display_h - 1))
        cv2.circle(canvas, (fx, fy), 10, (0, 255, 0), -1)
        cv2.putText(
            canvas,
            f"x={gx:.1f}  y={gy:.1f}  conf={gc:.2f}",
            (10, display_h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
        )
    else:
        cv2.putText(
            canvas,
            "gaze: no sample yet",
            (10, display_h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 140, 255),
            2,
        )

    return canvas


def calibrate_gaze_mapping(
    stream,
    gaze_estimator,
    detect_face_crop: Callable[[np.ndarray], Optional[np.ndarray]],
) -> Optional[dict[str, np.ndarray]]:
    """Collect gaze samples against fixed targets and fit an affine mapping."""
    if config.MOCK_MODE or not config.CALIBRATE_GAZE_EVERY_RUN:
        return None

    if not hasattr(gaze_estimator, "estimate"):
        return None

    logger.info("Starting gaze calibration …")
    print("[CALIB] Opening calibration window...")
    print("[CALIB] For each target: look at red dot, then press SPACE.")
    print("[CALIB] Press ESC anytime to skip calibration.")
    targets = build_gaze_calibration_targets()
    calib_w = min(config.SCREEN_WIDTH, 1280)
    calib_h = min(config.SCREEN_HEIGHT, 720)
    window = "Gaze Calibration"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, calib_w, calib_h)
    cv2.moveWindow(window, 30, 30)
    try:
        cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass

    samples: list[tuple[float, float, float, float]] = []
    cancelled = False
    for idx, (target_x_norm, target_y_norm) in enumerate(targets, start=1):
        target_x = int(target_x_norm * calib_w)
        target_y = int(target_y_norm * calib_h)
        print(f"[CALIB] Point {idx}/{len(targets)} ready. Press SPACE to capture.")

        while True:
            _, frame = stream.read()
            if frame is None:
                time.sleep(0.01)
                continue

            canvas = np.zeros((calib_h, calib_w, 3), dtype=np.uint8)
            step_x = max(calib_w // 8, 1)
            step_y = max(calib_h // 6, 1)
            for x in range(0, calib_w, step_x):
                cv2.line(canvas, (x, 0), (x, calib_h), (35, 35, 35), 1)
            for y in range(0, calib_h, step_y):
                cv2.line(canvas, (0, y), (calib_w, y), (35, 35, 35), 1)

            cv2.circle(canvas, (target_x, target_y), 12, (0, 0, 255), -1)
            cv2.putText(canvas, f"Calibration {idx}/{len(targets)}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(canvas, "Look at red dot, then press SPACE", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(canvas, "ESC = skip calibration", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)

            cv2.imshow(window, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                cancelled = True
                break
            if key == 32:
                break

        if cancelled:
            break

        point_start = time.time()
        point_samples = 0

        while time.time() - point_start < config.GAZE_CALIBRATION_SECONDS_PER_POINT:
            _, frame = stream.read()
            if frame is None:
                time.sleep(0.01)
                continue

            face_crop = detect_face_crop(frame)
            canvas = np.zeros((calib_h, calib_w, 3), dtype=np.uint8)

            step_x = max(calib_w // 8, 1)
            step_y = max(calib_h // 6, 1)
            for x in range(0, calib_w, step_x):
                cv2.line(canvas, (x, 0), (x, calib_h), (35, 35, 35), 1)
            for y in range(0, calib_h, step_y):
                cv2.line(canvas, (0, y), (calib_w, y), (35, 35, 35), 1)

            cv2.circle(canvas, (target_x, target_y), 12, (0, 0, 255), -1)
            cv2.putText(canvas, f"Calibration {idx}/{len(targets)}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(canvas, "Look at the red dot", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            remaining = max(0.0, config.GAZE_CALIBRATION_SECONDS_PER_POINT - (time.time() - point_start))
            cv2.putText(canvas, f"capturing: {remaining:.1f}s", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            if face_crop is not None:
                gaze_result = gaze_estimator.estimate(face_crop)
                if gaze_result is not None:
                    samples.append(
                        (
                            float(gaze_result.gaze_vector[0]),
                            float(gaze_result.gaze_vector[1]),
                            float(target_x_norm * config.SCREEN_WIDTH),
                            float(target_y_norm * config.SCREEN_HEIGHT),
                        )
                    )
                    point_samples += 1
                    cv2.circle(canvas, (target_x, target_y), 25, (0, 255, 0), 2)

            cv2.imshow(window, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                cancelled = True
                break

        if cancelled:
            break

        logger.info("Calibration point %d/%d collected (%d samples)", idx, len(targets), point_samples)
        print(f"[CALIB] Point {idx}/{len(targets)} collected ({point_samples} samples).")

    cv2.destroyWindow(window)

    if cancelled:
        logger.warning("Calibration cancelled by user; using default mapping")
        print("[CALIB] Cancelled. Using default mapping.")
        return None

    calibration = fit_gaze_calibration(samples)
    if calibration is None:
        logger.warning("Gaze calibration failed; using default mapping")
        print("[CALIB] Failed (not enough samples). Using default mapping.")
        return None

    logger.info("Gaze calibration completed with %d samples", len(samples))
    print(f"[CALIB] Completed with {len(samples)} samples.")
    return calibration
