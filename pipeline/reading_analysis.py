from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from ocr_module.screen_email_ocr import OCRWord
from pipeline.gaze_input import GazeSample
from pipeline.phishing_analysis import PhishingAnalysisResult
from pipeline.word_lookup import WordLocator


@dataclass
class PhraseReadingStatus:
    phrase: str
    category: str
    matched_words: list[str]
    read_words: list[str]
    coverage: float
    status: str


@dataclass
class ReadingAnalysisResult:
    total_samples: int
    matched_samples: int
    words_with_hits: list[str]
    words_read_in_order: list[str]
    user_read_mail: str
    read_word_indices: list[int]
    word_dwell_times: dict[int, float]
    phrase_statuses: list[PhraseReadingStatus]

    @property
    def missed_phrases(self) -> list[PhraseReadingStatus]:
        return [status for status in self.phrase_statuses if status.status != "fully_read"]


def _normalize_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _phrase_token_matches(words: list[OCRWord], phrase_text: str) -> list[int]:
    target_tokens = [_normalize_token(token) for token in phrase_text.split()]
    target_tokens = [token for token in target_tokens if token]
    if not target_tokens:
        return []

    normalized_words = [_normalize_token(word.text) for word in words]
    best_match: list[int] = []

    for start in range(len(words) - len(target_tokens) + 1):
        window = normalized_words[start : start + len(target_tokens)]
        if window == target_tokens:
            return list(range(start, start + len(target_tokens)))

    for idx, token in enumerate(normalized_words):
        if token in target_tokens:
            best_match.append(idx)
    return best_match


def analyze_reading(
    words: list[OCRWord],
    gaze_samples: list[GazeSample],
    phishing_result: PhishingAnalysisResult,
    min_read_time_s: float = 0.02,
    padding_px: int = 8,
    max_gap_s: float = 1.0,
) -> ReadingAnalysisResult:
    locator = WordLocator(words, padding_px=padding_px)
    matched_samples = 0
    first_hits: dict[int, float] = {}
    dwell_times: dict[int, float] = defaultdict(float)
    sample_hits: list[int] = []
    last_index: int | None = None
    last_timestamp: float | None = None

    for sample in gaze_samples:
        match = locator.find(sample.x, sample.y)
        if match is None:
            last_index = None
            last_timestamp = sample.timestamp_s
            continue

        matched_samples += 1
        sample_hits.append(match.index)
        first_hits.setdefault(match.index, sample.timestamp_s)

        if last_index == match.index and last_timestamp is not None:
            dt = sample.timestamp_s - last_timestamp
            if 0.0 <= dt <= max_gap_s:
                dwell_times[match.index] += dt

        last_index = match.index
        last_timestamp = sample.timestamp_s

    read_word_indices = sorted(
        [index for index, dwell in dwell_times.items() if dwell >= min_read_time_s],
        key=lambda index: first_hits[index],
    )
    read_word_index_set = set(read_word_indices)

    phrase_statuses: list[PhraseReadingStatus] = []
    for phrase in phishing_result.suspicious_phrases:
        matched_indices = _phrase_token_matches(words, phrase.text)
        matched_words = [words[index].text for index in matched_indices]
        read_words = [words[index].text for index in matched_indices if index in read_word_index_set]
        coverage = (len(read_words) / len(matched_words)) if matched_words else 0.0
        if coverage >= 0.999:
            status = "fully_read"
        elif coverage > 0.0:
            status = "partially_read"
        else:
            status = "unread"

        phrase_statuses.append(
            PhraseReadingStatus(
                phrase=phrase.text,
                category=phrase.category,
                matched_words=matched_words,
                read_words=read_words,
                coverage=round(coverage, 3),
                status=status,
            )
        )

    read_words_in_email_order = [word.text for idx, word in enumerate(words) if idx in read_word_index_set]
    words_with_hits = [words[index].text for index in sorted(first_hits, key=lambda idx: first_hits[idx])]

    return ReadingAnalysisResult(
        total_samples=len(gaze_samples),
        matched_samples=matched_samples,
        words_with_hits=words_with_hits,
        words_read_in_order=read_words_in_email_order,
        user_read_mail=" ".join(read_words_in_email_order),
        read_word_indices=read_word_indices,
        word_dwell_times=dict(sorted(dwell_times.items())),
        phrase_statuses=phrase_statuses,
    )

