"""Map Neon export coordinates into the existing screen-space contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import config

from .export_parser import NeonFixationSample, NeonGazeSample


@dataclass
class NeonMappedSample:
    timestamp_s: float
    screen_x: float
    screen_y: float
    confidence: float


class NeonSignalMapper:
    """Project Neon gaze coordinates into the screen-space used downstream.

    For the first implementation this is intentionally lightweight. The
    default mode assumes the export already contains coordinates in the space
    you want to consume downstream. If your recordings are still in scene-camera
    space, plug in a transformation later (e.g. from a mapped export or a
    homography computed from reference images / markers).
    """

    def __init__(self, coordinate_mode: Optional[str] = None) -> None:
        self.coordinate_mode = coordinate_mode or config.NEON_COORDINATE_MODE

    def gaze_to_screen(self, sample: NeonGazeSample) -> NeonMappedSample:
        screen_x, screen_y = self._project_xy(sample.gaze_x_px, sample.gaze_y_px)
        return NeonMappedSample(
            timestamp_s=sample.timestamp_ns / 1_000_000_000.0,
            screen_x=screen_x,
            screen_y=screen_y,
            confidence=sample.confidence,
        )

    def fixation_to_screen(self, sample: NeonFixationSample) -> Tuple[float, float]:
        return self._project_xy(sample.fixation_x_px, sample.fixation_y_px)

    def _project_xy(self, x: float, y: float) -> Tuple[float, float]:
        if self.coordinate_mode == "normalize" and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            return x * config.SCREEN_WIDTH, y * config.SCREEN_HEIGHT
        return x, y
