"""Derive a cognitive state from Neon eye-state and IMU features.

This is a lightweight rules-first adapter so the pipeline can start using
Neon-derived signals immediately. It can be replaced later with a trained
classifier on the same feature contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Dict, List, Optional, Sequence, Tuple

import config

from .export_parser import NeonBlinkSample, NeonIMUSample


COGNITIVE_STATES = ("focused", "confused", "stressed", "neutral")


@dataclass
class NeonCognitiveFeatures:
    avg_eye_openness: float
    blink_rate: float
    head_motion_std: float
    pupil_variability: float
    openness_variability: float


def _pick_float(row: Dict[str, str], *keys: str) -> Optional[float]:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            try:
                return float(value)
            except ValueError:
                continue
    return None


class NeonCognitiveAdapter:
    def __init__(self, window_s: float = config.NEON_COG_FEATURE_WINDOW_S) -> None:
        self.window_s = window_s

    def predict_from_window(
        self,
        eye_state_rows: Sequence[Dict[str, str]],
        imu_samples: Sequence[NeonIMUSample],
        blinks: Sequence[NeonBlinkSample],
    ) -> Tuple[str, Dict[str, float], NeonCognitiveFeatures]:
        if not eye_state_rows and not imu_samples and not blinks:
            neutral_conf = {state: 0.25 for state in COGNITIVE_STATES}
            return "neutral", neutral_conf, NeonCognitiveFeatures(0.0, 0.0, 0.0, 0.0, 0.0)

        openness_values: List[float] = []
        pupil_values: List[float] = []
        for row in eye_state_rows:
            openness = _pick_float(
                row,
                "eyelid_aperture_left_mm",
                "eyelid_aperture_right_mm",
                "eyelid_angle_top_left",
                "eyelid_angle_bottom_left",
                "eyelid_angle_top_right",
                "eyelid_angle_bottom_right",
            )
            if openness is not None:
                openness_values.append(abs(openness))

            pupil = _pick_float(row, "pupil_diameter_left_mm", "pupil_diameter_right_mm")
            if pupil is not None:
                pupil_values.append(pupil)

        if openness_values:
            openness_peak = max(max(openness_values), 1e-6)
            avg_eye_openness = min(mean(openness_values) / openness_peak, 1.0)
            openness_variability = pstdev(openness_values) if len(openness_values) > 1 else 0.0
        else:
            avg_eye_openness = 0.5
            openness_variability = 0.0

        pupil_variability = (pstdev(pupil_values) / max(mean(pupil_values), 1e-6)) if len(pupil_values) > 1 else 0.0

        if len(eye_state_rows) >= 2:
            start_ts = _pick_float(eye_state_rows[0], "timestamp_ns", "timestamp") or 0.0
            end_ts = _pick_float(eye_state_rows[-1], "timestamp_ns", "timestamp") or start_ts
            window_s = max((end_ts - start_ts) / 1_000_000_000.0, self.window_s)
        else:
            window_s = self.window_s

        blink_rate = len(blinks) / max(window_s, 1e-6)

        pitch_values = [s.pitch_deg for s in imu_samples if s.pitch_deg is not None]
        yaw_values = [s.yaw_deg for s in imu_samples if s.yaw_deg is not None]
        roll_values = [s.roll_deg for s in imu_samples if s.roll_deg is not None]
        motion_components = [pstdev(values) for values in (pitch_values, yaw_values, roll_values) if len(values) > 1]
        head_motion_std = mean(motion_components) if motion_components else 0.0

        features = NeonCognitiveFeatures(
            avg_eye_openness=round(avg_eye_openness, 3),
            blink_rate=round(blink_rate, 3),
            head_motion_std=round(head_motion_std, 3),
            pupil_variability=round(pupil_variability, 3),
            openness_variability=round(openness_variability, 3),
        )

        scores: Dict[str, float] = {state: 0.0 for state in COGNITIVE_STATES}

        if features.avg_eye_openness >= 0.75:
            scores["focused"] += 0.8
        if 0.08 <= features.blink_rate <= 0.5:
            scores["focused"] += 0.7
        if features.head_motion_std < 5.0:
            scores["focused"] += 0.5

        if features.head_motion_std >= 7.0:
            scores["confused"] += 0.8
        if 0.4 <= features.blink_rate <= 0.8:
            scores["confused"] += 0.5
        if features.openness_variability >= 0.2:
            scores["confused"] += 0.4

        if features.blink_rate > 0.75:
            scores["stressed"] += 0.9
        if features.avg_eye_openness < 0.4:
            scores["stressed"] += 0.8
        if features.pupil_variability > 0.2:
            scores["stressed"] += 0.4

        scores["neutral"] += 0.3

        total = sum(scores.values()) + 1e-8
        confidence = {state: round(score / total, 3) for state, score in scores.items()}
        state = max(confidence, key=confidence.get)
        return state, confidence, features
