"""
gaze_module/mediapipe_fallback.py – Gaze approximation via MediaPipe Face Mesh.

When the L2CS-Net pretrained weights are unavailable this module provides a
lightweight alternative using MediaPipe's 468-landmark Face Mesh.

Approach:
    1. Detect the face mesh in each frame.
    2. Use the iris landmarks (indices 468-477) to derive a rough gaze
       direction based on iris displacement relative to the eye corners.
    3. Map the normalised gaze offset to screen coordinates.

Accuracy is lower than L2CS-Net but requires no GPU and no external weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

import config
from utils.logging_utils import get_logger

logger = get_logger(__name__)

try:
    import mediapipe as mp  # type: ignore
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False
    logger.warning("mediapipe not installed – MediaPipe gaze fallback disabled")


# ──────────────────────────────────────────────
# Relevant landmark indices (MediaPipe Face Mesh)
# ──────────────────────────────────────────────
# Left eye corners
_L_EYE_INNER = 133
_L_EYE_OUTER = 33
# Right eye corners
_R_EYE_INNER = 362
_R_EYE_OUTER = 263
# Iris centre approximations (refined iris model, indices 468+)
_L_IRIS_CENTER = 468
_R_IRIS_CENTER = 473


@dataclass
class MPGazeResult:
    """Output of the MediaPipe gaze estimator."""
    gaze_x_norm: float        # –1 (left) … +1 (right)
    gaze_y_norm: float        # –1 (up)   … +1 (down)
    screen_x: float
    screen_y: float
    confidence: float

    def to_dict(self) -> dict:
        return {
            "gaze_x_norm": self.gaze_x_norm,
            "gaze_y_norm": self.gaze_y_norm,
            "screen_x": self.screen_x,
            "screen_y": self.screen_y,
            "confidence": self.confidence,
        }


class MediaPipeGazeEstimator:
    """Estimate gaze using MediaPipe Face Mesh iris tracking."""

    def __init__(self) -> None:
        if not _MP_AVAILABLE:
            raise RuntimeError("mediapipe is required for MediaPipeGazeEstimator")

        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,   # enable iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        logger.info("MediaPipeGazeEstimator initialised")

    def estimate(self, frame: np.ndarray) -> Optional[MPGazeResult]:
        """Run gaze estimation on a BGR frame.

        Returns ``None`` if no face is detected.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        lm = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape

        # Iris & eye corner pixel positions
        l_iris = np.array([lm[_L_IRIS_CENTER].x * w, lm[_L_IRIS_CENTER].y * h])
        r_iris = np.array([lm[_R_IRIS_CENTER].x * w, lm[_R_IRIS_CENTER].y * h])

        l_inner = np.array([lm[_L_EYE_INNER].x * w, lm[_L_EYE_INNER].y * h])
        l_outer = np.array([lm[_L_EYE_OUTER].x * w, lm[_L_EYE_OUTER].y * h])
        r_inner = np.array([lm[_R_EYE_INNER].x * w, lm[_R_EYE_INNER].y * h])
        r_outer = np.array([lm[_R_EYE_OUTER].x * w, lm[_R_EYE_OUTER].y * h])

        # Normalised iris offset within each eye (0 = outer, 1 = inner)
        l_eye_w = np.linalg.norm(l_inner - l_outer) + 1e-6
        r_eye_w = np.linalg.norm(r_inner - r_outer) + 1e-6

        l_ratio_x = (l_iris[0] - l_outer[0]) / l_eye_w
        r_ratio_x = (r_iris[0] - r_outer[0]) / r_eye_w
        gaze_x = float(np.mean([l_ratio_x, r_ratio_x]) * 2 - 1)  # –1 … +1

        l_eye_centre_y = (l_inner[1] + l_outer[1]) / 2
        r_eye_centre_y = (r_inner[1] + r_outer[1]) / 2
        l_ratio_y = (l_iris[1] - l_eye_centre_y) / (l_eye_w * 0.5 + 1e-6)
        r_ratio_y = (r_iris[1] - r_eye_centre_y) / (r_eye_w * 0.5 + 1e-6)
        gaze_y = float(np.mean([l_ratio_y, r_ratio_y]))            # –1 … +1

        # Map to screen
        sx = config.SCREEN_WIDTH / 2 + gaze_x * (config.SCREEN_WIDTH / 2)
        sy = config.SCREEN_HEIGHT / 2 + gaze_y * (config.SCREEN_HEIGHT / 2)
        sx = float(np.clip(sx, 0, config.SCREEN_WIDTH))
        sy = float(np.clip(sy, 0, config.SCREEN_HEIGHT))

        # Rough confidence from detection quality
        conf = float(min(lm[_L_IRIS_CENTER].visibility or 0.5, 1.0))

        return MPGazeResult(
            gaze_x_norm=round(gaze_x, 4),
            gaze_y_norm=round(gaze_y, 4),
            screen_x=round(sx, 1),
            screen_y=round(sy, 1),
            confidence=round(conf, 3),
        )

    def close(self) -> None:
        self._face_mesh.close()
