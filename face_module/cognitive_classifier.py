"""
face_module/cognitive_classifier.py – Map facial behaviour features to a
discrete cognitive / attention state.

States: ["focused", "confused", "stressed", "neutral"]

Two modes of operation:

1. **Pretrained classifier** – load a scikit-learn model saved with joblib
   (path: config.COG_CLASSIFIER_PATH).
2. **Rule-based heuristic** – if no trained model is available, a simple
   hand-crafted rule set is used.  This is useful for bootstrapping before
   labelled data is collected.

A training helper ``train_classifier()`` is included so you can fit a
RandomForest on labelled feature windows and serialise it to disk.
"""

from __future__ import annotations

import os
from collections import deque
from typing import Dict, List, Optional, Tuple

import joblib  # type: ignore
import numpy as np
from sklearn.ensemble import RandomForestClassifier  # type: ignore

import config
from utils.logging_utils import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
COGNITIVE_STATES = ["focused", "confused", "stressed", "neutral"]

# Features used by the classifier (order matters for the sklearn model)
CLASSIFIER_FEATURES = [
    "avg_eye_openness",
    "blink_rate",          # blinks per second in the window
    "head_pitch_std",
    "head_yaw_std",
    "head_roll_std",
    "AU04_mean",           # brow lowerer
    "AU45_mean",           # blink AU
]


# ──────────────────────────────────────────────
# Feature aggregation over a time window
# ──────────────────────────────────────────────
class FeatureAggregator:
    """Buffer per-frame features and compute window-level statistics."""

    def __init__(self, window_s: float = config.COG_FEATURE_WINDOW_S) -> None:
        self.window_s = window_s
        self._buffer: deque[Dict[str, float]] = deque()

    def push(self, frame_features: Dict[str, float]) -> None:
        """Add one frame's extracted features."""
        self._buffer.append(frame_features)
        # Evict stale samples
        cutoff = frame_features.get("timestamp", 0) - self.window_s
        while self._buffer and self._buffer[0].get("timestamp", 0) < cutoff:
            self._buffer.popleft()

    def aggregate(self) -> Optional[np.ndarray]:
        """Return a feature vector (shape ``(len(CLASSIFIER_FEATURES),)``)
        aggregated over the current window, or *None* if buffer is empty."""
        if not self._buffer:
            return None

        buf = list(self._buffer)
        n = len(buf)

        eye_open = np.mean(
            [(f.get("left_eye_openness", 0) + f.get("right_eye_openness", 0)) / 2 for f in buf]
        )
        blinks = sum(1 for f in buf if f.get("blink_detected", 0) > 0.5)
        dur = max(buf[-1].get("timestamp", 0) - buf[0].get("timestamp", 0), 0.1)
        blink_rate = blinks / dur

        pitches = [f.get("head_pitch", 0) for f in buf]
        yaws    = [f.get("head_yaw", 0) for f in buf]
        rolls   = [f.get("head_roll", 0) for f in buf]

        au04 = np.mean([f.get("AU04_r", 0) for f in buf])
        au45 = np.mean([f.get("AU45_r", 0) for f in buf])

        vec = np.array([
            eye_open,
            blink_rate,
            np.std(pitches),
            np.std(yaws),
            np.std(rolls),
            au04,
            au45,
        ], dtype=np.float32)

        return vec


# ──────────────────────────────────────────────
# Cognitive State Classifier
# ──────────────────────────────────────────────
class CognitiveClassifier:
    """Predict cognitive state from aggregated facial features.

    Falls back to a rule-based heuristic if no sklearn model is available.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model: Optional[RandomForestClassifier] = None
        mp = model_path or config.COG_CLASSIFIER_PATH
        if os.path.isfile(mp) and not config.MOCK_MODE:
            try:
                self.model = joblib.load(mp)
                logger.info("Loaded cognitive classifier from %s", mp)
            except Exception as exc:
                logger.warning("Failed to load classifier: %s", exc)

        if self.model is None:
            logger.info("Using rule-based cognitive state heuristic")

        self.aggregator = FeatureAggregator()

    # ── per-frame update ─────────────────────────
    def update(self, frame_features: Dict[str, float]) -> None:
        """Feed a single frame's features into the aggregation buffer."""
        self.aggregator.push(frame_features)

    # ── predict ──────────────────────────────────
    def predict(self) -> Tuple[str, Dict[str, float]]:
        """Return (state, confidence_dict).

        ``confidence_dict`` maps each state to a probability-like score.
        """
        vec = self.aggregator.aggregate()
        if vec is None:
            return "neutral", {s: 0.25 for s in COGNITIVE_STATES}

        if self.model is not None:
            return self._predict_ml(vec)
        return self._predict_rules(vec)

    # ── ML prediction ────────────────────────────
    def _predict_ml(self, vec: np.ndarray) -> Tuple[str, Dict[str, float]]:
        X = vec.reshape(1, -1)
        proba = self.model.predict_proba(X)[0]
        classes = list(self.model.classes_)
        conf = {c: round(float(p), 3) for c, p in zip(classes, proba)}
        state = classes[int(np.argmax(proba))]
        return state, conf

    # ── Rule-based heuristic ─────────────────────
    @staticmethod
    def _predict_rules(vec: np.ndarray) -> Tuple[str, Dict[str, float]]:
        """Simple hand-crafted rules.  Replace with a trained model later.

        Feature order (see CLASSIFIER_FEATURES):
            0: avg_eye_openness
            1: blink_rate
            2: head_pitch_std
            3: head_yaw_std
            4: head_roll_std
            5: AU04_mean (brow lowerer)
            6: AU45_mean (blink)
        """
        eye_open   = vec[0]
        blink_rate = vec[1]
        pitch_std  = vec[2]
        yaw_std    = vec[3]
        au04       = vec[5]

        scores: Dict[str, float] = {s: 0.0 for s in COGNITIVE_STATES}

        # Focused: steady gaze, normal blink, brow neutral
        if pitch_std < 0.05 and yaw_std < 0.05 and 0.1 < blink_rate < 0.5:
            scores["focused"] += 1.0
        # Confused: head tilting, furrowed brow
        if yaw_std > 0.08 or au04 > 1.5:
            scores["confused"] += 1.0
        # Stressed: high blink rate, tense brow
        if blink_rate > 0.5 and au04 > 1.0:
            scores["stressed"] += 1.0
        # Neutral otherwise
        scores["neutral"] += 0.3

        total = sum(scores.values()) + 1e-8
        conf = {s: round(v / total, 3) for s, v in scores.items()}
        state = max(conf, key=conf.get)  # type: ignore[arg-type]
        return state, conf


# ──────────────────────────────────────────────
# Training helper
# ──────────────────────────────────────────────
def train_classifier(
    X: np.ndarray,
    y: List[str],
    save_path: Optional[str] = None,
) -> RandomForestClassifier:
    """Train a RandomForest on labelled feature windows and save to disk.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, 7)
        Each row is an aggregated feature vector (see CLASSIFIER_FEATURES).
    y : list[str]
        Labels from COGNITIVE_STATES.
    save_path : str, optional
        Where to save the model; defaults to config.COG_CLASSIFIER_PATH.

    Returns
    -------
    RandomForestClassifier
    """
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    clf.fit(X, y)
    logger.info("Trained cognitive classifier – accuracy on train: %.3f", clf.score(X, y))

    sp = save_path or config.COG_CLASSIFIER_PATH
    os.makedirs(os.path.dirname(sp), exist_ok=True)
    joblib.dump(clf, sp)
    logger.info("Saved classifier to %s", sp)
    return clf
