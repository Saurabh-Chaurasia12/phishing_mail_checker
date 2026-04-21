from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Optional

from ocr_module.screen_email_ocr import OCRWord


@dataclass(frozen=True)
class WordMatch:
    index: int
    word: OCRWord


class WordLocator:
    """Locate OCR words for screen-space gaze coordinates."""

    def __init__(self, words: list[OCRWord], padding_px: int = 8) -> None:
        indexed = list(enumerate(words))
        self._entries = sorted(indexed, key=lambda item: item[1].left)
        self._starts = [word.left for _, word in self._entries]
        self._padding = padding_px
        self._max_width = max((word.width for word in words), default=0)

    def find(self, x: float, y: float) -> Optional[WordMatch]:
        search_limit = bisect.bisect_right(self._starts, int(x) + self._padding)
        lower_bound = int(x) - self._max_width - self._padding

        for pos in range(search_limit - 1, -1, -1):
            original_index, word = self._entries[pos]
            if word.left < lower_bound:
                break
            if word.contains_point(x, y, padding=self._padding):
                return WordMatch(index=original_index, word=word)
        return None

