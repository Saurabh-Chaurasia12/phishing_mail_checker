"""
face_module/openface_extractor.py – Extract facial behaviour features using
the OpenFace toolkit (or a MediaPipe-based fallback).

OpenFace provides:
    • 68 facial landmarks
    • Eye openness (AU45 – blink)
    • Head pose (Rx, Ry, Rz)
    • 17 Facial Action Units (intensity + presence)

When OpenFace is not installed a *lightweight MediaPipe-based fallback*
is used that approximates a subset of these features:
    eye_openness, blink, head_pitch, head_yaw, head_roll.

All extractors conform to the same output dict schema so downstream
modules are agnostic to the backend.
"""

from __future__ import annotations

import csv
import os
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

import config
from utils.logging_utils import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────
# Output feature schema (common for all backends)
# ──────────────────────────────────────────────
FEATURE_KEYS = [
    "timestamp",
    # Eye
    "left_eye_openness",
    "right_eye_openness",
    "blink_detected",
    # Head pose (radians)
    "head_pitch",
    "head_yaw",
    "head_roll",
    # Action Units (intensity 0-5)
    "AU01_r",  # Inner brow raiser
    "AU02_r",  # Outer brow raiser
    "AU04_r",  # Brow lowerer
    "AU05_r",  # Upper lid raiser
    "AU06_r",  # Cheek raiser
    "AU07_r",  # Lid tightener
    "AU09_r",  # Nose wrinkler
    "AU10_r",  # Upper lip raiser
    "AU12_r",  # Lip corner puller
    "AU14_r",  # Dimpler
    "AU15_r",  # Lip corner depressor
    "AU17_r",  # Chin raiser
    "AU20_r",  # Lip stretcher
    "AU23_r",  # Lip tightener
    "AU25_r",  # Lips part
    "AU26_r",  # Jaw drop
    "AU45_r",  # Blink
]


def _empty_features() -> Dict[str, float]:
    return {k: 0.0 for k in FEATURE_KEYS}


# ══════════════════════════════════════════════
# Backend 1: OpenFace CLI wrapper
# ══════════════════════════════════════════════
class OpenFaceExtractor:
    """Call the OpenFace ``FeatureExtraction`` binary per-frame.

    For real-time use the binary is invoked on short video clips or
    individual images written to a temp directory.
    """

    def __init__(self, binary_path: str = config.OPENFACE_BINARY) -> None:
        self.binary = binary_path
        if not os.path.isfile(self.binary):
            raise FileNotFoundError(
                f"OpenFace binary not found at {self.binary}. "
                "Set OPENFACE_BIN env variable or install OpenFace."
            )
        logger.info("OpenFaceExtractor ready (binary=%s)", self.binary)

    def extract_from_frame(self, frame: np.ndarray) -> Dict[str, float]:
        """Write *frame* to a temp image, invoke OpenFace, parse CSV output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "frame.png")
            cv2.imwrite(img_path, frame)

            out_dir = os.path.join(tmpdir, "out")
            os.makedirs(out_dir, exist_ok=True)

            cmd = [
                self.binary,
                "-f", img_path,
                "-out_dir", out_dir,
                "-aus",       # Action Units
                "-pose",      # Head pose
                "-2Dfp",      # 2-D landmarks
                "-q",         # quiet
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True, timeout=10)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                logger.warning("OpenFace invocation failed: %s", exc)
                return _empty_features()

            # Parse the CSV produced by OpenFace
            csv_files = [f for f in os.listdir(out_dir) if f.endswith(".csv")]
            if not csv_files:
                return _empty_features()

            csv_path = os.path.join(out_dir, csv_files[0])
            return self._parse_csv(csv_path)

    @staticmethod
    def _parse_csv(path: str) -> Dict[str, float]:
        """Parse OpenFace output CSV and map to our feature schema."""
        features = _empty_features()
        features["timestamp"] = time.time()

        with open(path, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:  # usually a single row for one image
                # Strip whitespace from keys
                row = {k.strip(): v.strip() for k, v in row.items()}

                # Head pose
                features["head_pitch"] = float(row.get("pose_Rx", 0))
                features["head_yaw"]   = float(row.get("pose_Ry", 0))
                features["head_roll"]  = float(row.get("pose_Rz", 0))

                # Action units
                for au_key in FEATURE_KEYS:
                    if au_key.startswith("AU") and au_key in row:
                        features[au_key] = float(row[au_key])

                # Blink heuristic
                au45 = float(row.get("AU45_r", "0") or 0)
                features["AU45_r"] = au45
                features["blink_detected"] = 1.0 if au45 > 1.5 else 0.0

                # Eye openness from AU5 (upper lid raiser) & AU45
                features["left_eye_openness"]  = max(1.0 - au45 / 5.0, 0.0)
                features["right_eye_openness"] = max(1.0 - au45 / 5.0, 0.0)
                break  # first row only

        return features


# ══════════════════════════════════════════════
# Backend 2: MediaPipe Face Mesh fallback
# ══════════════════════════════════════════════
class MediaPipeFaceExtractor:
    """Approximate facial features using MediaPipe Face Mesh.

    Provides a subset of the full OpenFace feature set — sufficient for
    the cognitive state classifier to function.
    """

    # Landmark indices for eye aspect ratio (EAR)
    _L_EYE = [33, 160, 158, 133, 153, 144]
    _R_EYE = [362, 385, 387, 263, 373, 380]

    def __init__(self) -> None:
        try:
            import mediapipe as mp  # type: ignore
        except ImportError as exc:
            raise RuntimeError("mediapipe is required for the fallback face extractor") from exc

        solutions = getattr(mp, "solutions", None)
        if solutions is None or not hasattr(solutions, "face_mesh"):
            raise RuntimeError(
                "mediapipe.solutions.face_mesh is unavailable in the installed mediapipe package"
            )

        self._mp_face_mesh = solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._prev_ear: Optional[float] = None
        logger.info("MediaPipeFaceExtractor initialised (fallback mode)")

    def extract_from_frame(self, frame: np.ndarray) -> Dict[str, float]:
        """Extract approximate features from a BGR frame."""
        features = _empty_features()
        features["timestamp"] = time.time()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._mp_face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return features

        lm = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape

        # ── Eye Aspect Ratio (EAR) ──────────
        def ear(indices: List[int]) -> float:
            pts = np.array([[lm[i].x * w, lm[i].y * h] for i in indices])
            v1 = np.linalg.norm(pts[1] - pts[5])
            v2 = np.linalg.norm(pts[2] - pts[4])
            hz = np.linalg.norm(pts[0] - pts[3]) + 1e-6
            return float((v1 + v2) / (2.0 * hz))

        l_ear = ear(self._L_EYE)
        r_ear = ear(self._R_EYE)
        avg_ear = (l_ear + r_ear) / 2.0

        features["left_eye_openness"]  = round(l_ear, 4)
        features["right_eye_openness"] = round(r_ear, 4)

        # Blink detection via EAR threshold
        blink = avg_ear < 0.21
        features["blink_detected"] = 1.0 if blink else 0.0
        features["AU45_r"] = round(max(0, (0.3 - avg_ear) / 0.3) * 5, 2)

        # ── Head pose (rough, from nose tip + forehead) ──
        nose = np.array([lm[1].x, lm[1].y, lm[1].z])
        forehead = np.array([lm[10].x, lm[10].y, lm[10].z])
        chin = np.array([lm[152].x, lm[152].y, lm[152].z])
        left_ear_pt = np.array([lm[234].x, lm[234].y, lm[234].z])
        right_ear_pt = np.array([lm[454].x, lm[454].y, lm[454].z])

        features["head_pitch"] = round(float(nose[1] - forehead[1]) * 3.14, 3)
        features["head_yaw"]   = round(float(right_ear_pt[2] - left_ear_pt[2]) * 3.14, 3)
        features["head_roll"]  = round(float(right_ear_pt[1] - left_ear_pt[1]) * 3.14, 3)

        # Approximate some AUs from geometry
        brow_inner_l = np.array([lm[65].y, lm[65].z])
        brow_inner_r = np.array([lm[295].y, lm[295].z])
        features["AU04_r"] = round(abs(float(brow_inner_l[0] - nose[1])) * 10, 2)  # brow lowerer proxy

        self._prev_ear = avg_ear
        return features

    def close(self) -> None:
        self._mp_face_mesh.close()


# ══════════════════════════════════════════════
# Mock extractor for testing
# ══════════════════════════════════════════════
class MockFaceExtractor:
    """Return random plausible features.  Zero dependencies."""

    def extract_from_frame(self, frame: np.ndarray) -> Dict[str, float]:
        features = _empty_features()
        features["timestamp"] = time.time()
        features["left_eye_openness"]  = float(np.random.uniform(0.2, 0.4))
        features["right_eye_openness"] = float(np.random.uniform(0.2, 0.4))
        features["blink_detected"]     = float(np.random.choice([0, 1], p=[0.9, 0.1]))
        features["head_pitch"] = float(np.random.normal(0, 0.1))
        features["head_yaw"]   = float(np.random.normal(0, 0.1))
        features["head_roll"]  = float(np.random.normal(0, 0.05))
        features["AU04_r"]     = float(np.random.uniform(0, 2))
        features["AU45_r"]     = float(np.random.uniform(0, 1))
        return features

    def close(self) -> None:
        pass


# ══════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════
def create_face_extractor() -> OpenFaceExtractor | MediaPipeFaceExtractor | MockFaceExtractor:
    """Return the best available face extractor."""
    if config.MOCK_MODE:
        logger.info("Mock mode -> MockFaceExtractor")
        return MockFaceExtractor()

    if config.OPENFACE_AVAILABLE:
        try:
            return OpenFaceExtractor()
        except FileNotFoundError:
            pass

    # Try MediaPipe fallback
    try:
        return MediaPipeFaceExtractor()
    except RuntimeError:
        pass

    logger.warning("No face extractor available -> MockFaceExtractor")
    return MockFaceExtractor()
