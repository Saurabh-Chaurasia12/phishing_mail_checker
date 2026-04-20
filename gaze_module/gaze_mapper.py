"""
gaze_module/gaze_mapper.py – Map raw gaze points to email interface regions
and detect fixations.

Responsibilities:
    1. Accumulate gaze samples over time.
    2. Detect *fixations* (gaze stable within a radius for ≥ threshold time).
    3. Classify each fixation into an email region
       (sender, subject, body, url, attachments).
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

import config
from utils.logging_utils import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────
@dataclass
class GazeSample:
    """A single timestamped gaze observation."""
    timestamp: float
    screen_x: float
    screen_y: float
    confidence: float


@dataclass
class Fixation:
    """A detected fixation event."""
    centroid_x: float
    centroid_y: float
    duration_s: float
    start_time: float
    end_time: float
    region: Optional[str] = None  # mapped email region


@dataclass
class GazeMapperState:
    """Accumulated state returned by the mapper."""
    fixations: List[Fixation] = field(default_factory=list)
    region_dwell_times: Dict[str, float] = field(default_factory=dict)
    regions_seen: List[str] = field(default_factory=list)


# ──────────────────────────────────────────────
# Region classification helper
# ──────────────────────────────────────────────
def _point_in_region(
    x: float, y: float, regions: Dict[str, Tuple[int, int, int, int]]
) -> Optional[str]:
    """Return the region name that contains (x, y), or None."""
    for name, (x1, y1, x2, y2) in regions.items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            return name
    return None


# ──────────────────────────────────────────────
# GazeMapper
# ──────────────────────────────────────────────
class GazeMapper:
    """Accumulate gaze samples, detect fixations, and map them to email regions.

    Parameters
    ----------
    fixation_radius_px : float
        Maximum dispersion (Euclidean px) for a cluster to count as a fixation.
    fixation_min_duration_s : float
        Minimum dwell time in seconds.
    email_regions : dict
        Mapping of region name → (x1, y1, x2, y2).
    """

    def __init__(
        self,
        fixation_radius_px: float = config.FIXATION_RADIUS_PX,
        fixation_min_duration_s: float = config.FIXATION_MIN_DURATION_S,
        email_regions: Optional[Dict[str, Tuple[int, int, int, int]]] = None,
    ) -> None:
        self.fixation_radius = fixation_radius_px
        self.fixation_min_dur = fixation_min_duration_s
        self.email_regions = email_regions or config.EMAIL_REGIONS

        self._samples: List[GazeSample] = []
        self._fixations: List[Fixation] = []
        self._last_processed_sample_idx = 0
        self._last_emitted_fixation_end = 0.0

    # ── add sample ───────────────────────────────
    def add_sample(
        self,
        screen_x: float,
        screen_y: float,
        confidence: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """Record a new gaze sample."""
        self._samples.append(
            GazeSample(
                timestamp=timestamp or time.time(),
                screen_x=screen_x,
                screen_y=screen_y,
                confidence=confidence,
            )
        )

    # ── fixation detection (dispersion-based) ────
    def detect_fixations(self) -> List[Fixation]:
        """I-DT (dispersion-threshold) fixation detector.

        Scans the accumulated samples and returns any *new* fixations found
        since the last call.
        """
        new_fixations: List[Fixation] = []
        if len(self._samples) < 3:
            return new_fixations

        start_idx = max(self._last_processed_sample_idx - 1, 0)
        i = start_idx
        while i < len(self._samples):
            j = i + 1
            # Grow window while dispersion is within threshold
            while j < len(self._samples):
                xs = [s.screen_x for s in self._samples[i : j + 1]]
                ys = [s.screen_y for s in self._samples[i : j + 1]]
                dispersion = (max(xs) - min(xs)) + (max(ys) - min(ys))
                if dispersion > self.fixation_radius * 2:
                    break
                j += 1

            # Check duration
            window = self._samples[i:j]
            dur = window[-1].timestamp - window[0].timestamp
            if dur >= self.fixation_min_dur:
                cx = float(np.mean([s.screen_x for s in window]))
                cy = float(np.mean([s.screen_y for s in window]))
                region = _point_in_region(cx, cy, self.email_regions)
                fix = Fixation(
                    centroid_x=round(cx, 1),
                    centroid_y=round(cy, 1),
                    duration_s=round(dur, 3),
                    start_time=window[0].timestamp,
                    end_time=window[-1].timestamp,
                    region=region,
                )
                if fix.end_time > self._last_emitted_fixation_end:
                    new_fixations.append(fix)
                    self._last_emitted_fixation_end = fix.end_time
                i = j  # skip past this fixation
            else:
                i += 1

        self._fixations.extend(new_fixations)
        self._last_processed_sample_idx = len(self._samples)
        return new_fixations

    # ── summary ──────────────────────────────────
    def get_state(self) -> GazeMapperState:
        """Return an aggregated summary of all detected fixations."""
        dwell: Dict[str, float] = defaultdict(float)
        seen: set[str] = set()

        for f in self._fixations:
            if f.region:
                dwell[f.region] += f.duration_s
                seen.add(f.region)

        return GazeMapperState(
            fixations=list(self._fixations),
            region_dwell_times=dict(dwell),
            regions_seen=sorted(seen),
        )

    def reset(self) -> None:
        """Clear all accumulated samples and fixations."""
        self._samples.clear()
        self._fixations.clear()
