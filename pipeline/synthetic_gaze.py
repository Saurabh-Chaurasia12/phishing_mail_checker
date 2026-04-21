from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
from openpyxl import Workbook

from email_phishing_detector import (
    extract_email_ocr_bundle_from_image,
    resolve_tesseract_path,
    validate_tesseract,
)


@dataclass(frozen=True)
class SyntheticGazePoint:
    timestamp_s: float
    x: float
    y: float
    target_word: str


def _group_lines(words):
    words = sorted(words, key=lambda word: (word.top, word.left))
    lines: list[list] = []
    current: list = []
    current_top: int | None = None

    for word in words:
        if current_top is None or abs(word.top - current_top) <= 18:
            current.append(word)
            current_top = word.top if current_top is None else min(current_top, word.top)
        else:
            lines.append(sorted(current, key=lambda item: item.left))
            current = [word]
            current_top = word.top

    if current:
        lines.append(sorted(current, key=lambda item: item.left))
    return lines


def _point_for_word(word, rng: random.Random) -> tuple[float, float]:
    cx = word.left + word.width / 2.0
    cy = word.top + word.height / 2.0
    jitter_x = rng.uniform(-max(word.width * 0.18, 1.0), max(word.width * 0.18, 1.0))
    jitter_y = rng.uniform(-max(word.height * 0.18, 1.0), max(word.height * 0.18, 1.0))
    return round(cx + jitter_x, 2), round(cy + jitter_y, 2)


def _random_off_word_point(line, rng: random.Random) -> tuple[float, float]:
    left = min(word.left for word in line)
    right = max(word.left + word.width for word in line)
    top = min(word.top for word in line)
    bottom = max(word.top + word.height for word in line)
    x = rng.uniform(left - 45, right + 45)
    y = rng.uniform(top - 22, bottom + 22)
    return round(x, 2), round(y, 2)


def build_synthetic_gaze_from_image(
    image_path: str | Path,
    xlsx_out: str | Path,
    csv_out: str | Path,
    sample_interval_s: float = 0.01,
    seed: int = 7,
) -> dict[str, int]:
    rng = random.Random(seed)
    image_path = Path(image_path)
    xlsx_out = Path(xlsx_out)
    csv_out = Path(csv_out)

    resolve_tesseract_path(None)
    validate_tesseract()

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    bundle = extract_email_ocr_bundle_from_image(
        image,
        min_confidence=40.0,
        apply_threshold=True,
        psm=6,
        region_offset=(0, 0),
        crop_region=None,
    )
    words = sorted(bundle.words, key=lambda word: (word.top, word.left))
    if not words:
        raise ValueError("No OCR words detected.")

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "left", "top", "right", "bottom", "confidence"])
        for word in words:
            writer.writerow(
                [
                    word.text,
                    word.left,
                    word.top,
                    word.left + word.width,
                    word.top + word.height,
                    round(word.confidence, 3),
                ]
            )

    lines = _group_lines(words)
    samples: list[SyntheticGazePoint] = []
    timestamp = 1.0
    looked_word_indices: set[int] = set()

    for line_index, line in enumerate(lines):
        if not line:
            continue

        max_words = min(len(line), rng.randint(7, 14))
        cursor = 0
        words_seen_on_line = 0

        while cursor < len(line) and words_seen_on_line < max_words:
            word = line[cursor]

            # Some words are skipped entirely, especially short/common ones.
            skip_prob = 0.24 if len(word.text) <= 4 else 0.16
            if rng.random() < skip_prob:
                cursor += 1
                continue

            word_index = words.index(word)
            looked_word_indices.add(word_index)
            fixation_samples = rng.choices(
                population=[1, 2, 3, 4, 5, 6, 8, 10],
                weights=[8, 16, 22, 20, 13, 10, 7, 4],
                k=1,
            )[0]

            # Important-looking words get longer attention more often.
            lowered = word.text.lower()
            if any(token in lowered for token in ["register", "now", "contest", "april", "scholarships", "here"]):
                fixation_samples += rng.choice([1, 2, 3, 5])

            same_point = rng.random() < 0.35
            fixed_xy = _point_for_word(word, rng) if same_point else None
            for _ in range(fixation_samples):
                if fixed_xy is None:
                    x, y = _point_for_word(word, rng)
                else:
                    x, y = fixed_xy
                samples.append(
                    SyntheticGazePoint(
                        timestamp_s=round(timestamp, 3),
                        x=x,
                        y=y,
                        target_word=word.text,
                    )
                )
                timestamp += sample_interval_s

            words_seen_on_line += 1

            # Brief random glance around the line, sometimes not on any word.
            if rng.random() < 0.18:
                off_x, off_y = _random_off_word_point(line, rng)
                samples.append(
                    SyntheticGazePoint(
                        timestamp_s=round(timestamp, 3),
                        x=off_x,
                        y=off_y,
                        target_word="<random_glance>",
                    )
                )
                timestamp += sample_interval_s

            # Occasional regression to a previously read word on the same line.
            if cursor > 1 and rng.random() < 0.12:
                revisit = line[rng.randint(max(0, cursor - 3), cursor - 1)]
                revisit_xy = _point_for_word(revisit, rng)
                for _ in range(rng.choice([1, 2, 3])):
                    samples.append(
                        SyntheticGazePoint(
                            timestamp_s=round(timestamp, 3),
                            x=revisit_xy[0],
                            y=revisit_xy[1],
                            target_word=revisit.text,
                        )
                    )
                    timestamp += sample_interval_s

            cursor += rng.choice([1, 1, 1, 2])

        # Small pause / saccade between lines.
        timestamp += rng.uniform(0.03, 0.09)

        # Sometimes the user glances at the right-side image/logo area.
        if line_index < 6 and rng.random() < 0.22:
            x = round(rng.uniform(1180, 1410), 2)
            y = round(rng.uniform(110, 255), 2)
            samples.append(
                SyntheticGazePoint(
                    timestamp_s=round(timestamp, 3),
                    x=x,
                    y=y,
                    target_word="<logo_glance>",
                )
            )
            timestamp += sample_interval_s

    xlsx_out.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "gaze_samples"
    sheet.append(["time_stamp", "x", "y"])
    for sample in samples:
        sheet.append([sample.timestamp_s, sample.x, sample.y])
    workbook.save(xlsx_out)

    return {
        "ocr_words": len(words),
        "lines": len(lines),
        "samples": len(samples),
        "looked_words": len(looked_word_indices),
    }
