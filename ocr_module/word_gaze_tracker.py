"""
ocr_module/word_gaze_tracker.py – Map real-time gaze coordinates to OCR word
bounding boxes to determine which words the user is reading.

This module bridges the gaze module (screen-space x,y) with the OCR module
(word bounding boxes from the screen capture) to produce word-level reading
analytics:

    • Which words has the user looked at?
    • How long did they dwell on each word?
    • Did they notice suspicious keywords (URLs, urgency phrases)?
    • What fraction of the email have they read?
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import config
from ocr_module.screen_email_ocr import OCRWord
from utils.logging_utils import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────
@dataclass
class WordGazeSample:
    """A single gaze-to-word match."""
    word: str
    word_index: int        # index into the original OCRWord list
    timestamp: float
    screen_x: float
    screen_y: float


@dataclass
class WordReadingSummary:
    """Aggregated reading analytics from gaze-word tracking."""
    words_read: List[str]                                   # unique words gazed at (ordered by first look)
    word_dwell_times: Dict[str, float]                      # word → total dwell seconds
    suspicious_words_gazed: List[str]                       # suspicious keywords the user DID look at
    unread_suspicious_words: List[str]                      # suspicious keywords the user did NOT look at
    reading_coverage: float                                 # fraction of email words read (0.0–1.0)
    reading_sequence: List[Tuple[str, float]]               # (word, timestamp) in reading order
    total_words_in_email: int
    total_words_read: int
    total_gaze_samples: int

    def to_dict(self) -> dict:
        return {
            "words_read": self.words_read,
            "word_dwell_times": self.word_dwell_times,
            "suspicious_words_gazed": self.suspicious_words_gazed,
            "unread_suspicious_words": self.unread_suspicious_words,
            "reading_coverage": self.reading_coverage,
            "total_words_in_email": self.total_words_in_email,
            "total_words_read": self.total_words_read,
            "total_gaze_samples": self.total_gaze_samples,
        }


# ──────────────────────────────────────────────
# WordGazeTracker
# ──────────────────────────────────────────────
class WordGazeTracker:
    """Map gaze (x, y) to OCR word bounding boxes in real time.

    Parameters
    ----------
    ocr_words : list[OCRWord]
        Word bounding boxes from the screen capture (screen-absolute coordinates).
    gaze_padding_px : int
        Extra pixel padding around each word bbox for hit-testing.
    suspicious_keywords : list[str], optional
        Keywords to flag; defaults to ``config.SUSPICIOUS_KEYWORDS``.
    """

    def __init__(
        self,
        ocr_words: List[OCRWord],
        gaze_padding_px: int = 10,
        suspicious_keywords: Optional[List[str]] = None,
    ) -> None:
        self._words = ocr_words
        self._padding = gaze_padding_px
        self._suspicious_kw = [
            kw.lower() for kw in (suspicious_keywords or config.SUSPICIOUS_KEYWORDS)
        ]

        # Tracking state
        self._samples: List[WordGazeSample] = []
        self._words_first_seen: Dict[int, float] = {}      # word_index → first timestamp
        self._word_dwell: Dict[int, float] = defaultdict(float)  # word_index → total dwell (s)
        self._last_sample_time: Optional[float] = None
        self._last_word_idx: Optional[int] = None

        # Precompute which OCR words are suspicious
        self._suspicious_word_indices: set[int] = set()
        for i, w in enumerate(self._words):
            w_lower = w.text.lower()
            for kw in self._suspicious_kw:
                if kw in w_lower or w_lower in kw:
                    self._suspicious_word_indices.add(i)
                    break

        logger.info(
            "WordGazeTracker initialised: %d words, %d suspicious",
            len(self._words),
            len(self._suspicious_word_indices),
        )

    # ── Hit-test: which word is at (x, y)? ───────
    def map_gaze_to_word(self, screen_x: float, screen_y: float) -> Optional[OCRWord]:
        """Return the OCR word at screen position (x, y), or None."""
        for word in self._words:
            if word.contains_point(screen_x, screen_y, padding=self._padding):
                return word
        return None

    def _find_word_index(self, screen_x: float, screen_y: float) -> Optional[int]:
        """Return the index of the OCR word at (x, y), or None."""
        for i, word in enumerate(self._words):
            if word.contains_point(screen_x, screen_y, padding=self._padding):
                return i
        return None

    # ── Record a gaze sample ─────────────────────
    def add_gaze_sample(
        self,
        screen_x: float,
        screen_y: float,
        timestamp: Optional[float] = None,
    ) -> Optional[str]:
        """Record a gaze sample and return the word being gazed at (or None).

        The dwell time is computed as the delta from the previous sample
        (if it was on the same word).
        """
        ts = timestamp or time.time()
        word_idx = self._find_word_index(screen_x, screen_y)

        if word_idx is not None:
            word = self._words[word_idx]
            self._samples.append(
                WordGazeSample(
                    word=word.text,
                    word_index=word_idx,
                    timestamp=ts,
                    screen_x=screen_x,
                    screen_y=screen_y,
                )
            )

            # First time seeing this word
            if word_idx not in self._words_first_seen:
                self._words_first_seen[word_idx] = ts

            # Accumulate dwell time
            if (
                self._last_sample_time is not None
                and self._last_word_idx == word_idx
            ):
                dt = ts - self._last_sample_time
                if dt < 1.0:  # cap single-step dwell to avoid gaps inflating
                    self._word_dwell[word_idx] += dt

            self._last_sample_time = ts
            self._last_word_idx = word_idx
            return word.text

        # Gaze is not on any word
        self._last_sample_time = ts
        self._last_word_idx = None
        return None

    # ── Reading summary ──────────────────────────
    def get_reading_summary(self) -> WordReadingSummary:
        """Compute and return an aggregated reading summary."""
        # Unique words read, ordered by first look
        seen_indices = sorted(self._words_first_seen.keys(),
                              key=lambda i: self._words_first_seen[i])
        words_read = [self._words[i].text for i in seen_indices]

        # Dwell times (by word text, merge duplicates)
        dwell_by_text: Dict[str, float] = defaultdict(float)
        for idx, dt in self._word_dwell.items():
            dwell_by_text[self._words[idx].text] += dt
        word_dwell_times = {w: round(t, 3) for w, t in dwell_by_text.items()}

        # Suspicious words analysis
        gazed_suspicious_idx = self._suspicious_word_indices & set(self._words_first_seen.keys())
        unread_suspicious_idx = self._suspicious_word_indices - set(self._words_first_seen.keys())

        suspicious_words_gazed = list({self._words[i].text for i in gazed_suspicious_idx})
        unread_suspicious_words = list({self._words[i].text for i in unread_suspicious_idx})

        # Reading sequence
        reading_sequence = [(s.word, s.timestamp) for s in self._samples]

        # Coverage
        total_words = len(self._words)
        total_read = len(self._words_first_seen)
        coverage = total_read / max(total_words, 1)

        return WordReadingSummary(
            words_read=words_read,
            word_dwell_times=word_dwell_times,
            suspicious_words_gazed=suspicious_words_gazed,
            unread_suspicious_words=unread_suspicious_words,
            reading_coverage=round(coverage, 3),
            reading_sequence=reading_sequence,
            total_words_in_email=total_words,
            total_words_read=total_read,
            total_gaze_samples=len(self._samples),
        )

    def get_unread_suspicious_words(self) -> List[str]:
        """Return suspicious keywords the user has NOT gazed at."""
        unread_idx = self._suspicious_word_indices - set(self._words_first_seen.keys())
        return list({self._words[i].text for i in unread_idx})

    def reset(self) -> None:
        """Clear all tracking state."""
        self._samples.clear()
        self._words_first_seen.clear()
        self._word_dwell.clear()
        self._last_sample_time = None
        self._last_word_idx = None
