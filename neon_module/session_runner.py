"""Offline Neon session execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import config
from fusion_module.multimodal_fusion_agent import FusionAgent, FusionVerdict
from gaze_module.gaze_mapper import GazeMapper
from ocr_module.screen_email_ocr import OCRWord
from ocr_module.word_gaze_tracker import WordGazeTracker
from utils.logging_utils import get_logger

from .cognitive_adapter import NeonCognitiveAdapter
from .export_parser import NeonRecording, parse_recording
from .signal_mapper import NeonSignalMapper

logger = get_logger(__name__)


@dataclass
class NeonSessionResult:
    verdicts: List[FusionVerdict]
    recording: NeonRecording


def _slice_eye_rows(rows: List[Dict[str, str]], start_ns: int, end_ns: int) -> List[Dict[str, str]]:
    sliced: List[Dict[str, str]] = []
    for row in rows:
        ts_raw = row.get("timestamp_ns") or row.get("timestamp") or ""
        try:
            ts_ns = int(float(ts_raw))
        except ValueError:
            continue
        if start_ns <= ts_ns <= end_ns:
            sliced.append(row)
    return sliced


def _slice_inclusive(items, start_ns: int, end_ns: int, timestamp_attr: str):
    return [item for item in items if start_ns <= getattr(item, timestamp_attr) <= end_ns]


def run_neon_offline_session(
    recording_dir: str,
    phish_result: Any,
    ocr_words: Optional[List[OCRWord]] = None,
) -> NeonSessionResult:
    recording = parse_recording(recording_dir)
    if not recording.gaze_samples:
        raise RuntimeError(f"No gaze.csv rows found in Neon recording: {recording_dir}")

    gaze_mapper = GazeMapper(fixation_min_duration_s=config.NEON_FIXATION_MIN_DURATION_S)
    fusion_agent = FusionAgent()
    signal_mapper = NeonSignalMapper()
    cognitive_adapter = NeonCognitiveAdapter() if config.NEON_ENABLE_COGNITIVE else None
    word_tracker = WordGazeTracker(ocr_words) if ocr_words else None

    start_ns = recording.start_time_ns
    end_ns = recording.end_time_ns
    window_ns = max(int(config.FUSION_INTERVAL_S * 1_000_000_000), 1)

    verdicts: List[FusionVerdict] = []
    window_start = start_ns
    window_end = min(start_ns + window_ns, end_ns)

    gaze_index = 0
    total_gaze_samples = len(recording.gaze_samples)

    while window_start <= end_ns:
        while gaze_index < total_gaze_samples and recording.gaze_samples[gaze_index].timestamp_ns <= window_end:
            gaze_sample = recording.gaze_samples[gaze_index]
            mapped = signal_mapper.gaze_to_screen(gaze_sample)
            gaze_mapper.add_sample(
                mapped.screen_x,
                mapped.screen_y,
                mapped.confidence,
                timestamp=mapped.timestamp_s,
            )
            if word_tracker is not None:
                word_tracker.add_gaze_sample(mapped.screen_x, mapped.screen_y, timestamp=mapped.timestamp_s)
            gaze_index += 1

        gaze_mapper.detect_fixations()
        gaze_state = gaze_mapper.get_state()

        if cognitive_adapter is not None:
            window_eye_rows = _slice_eye_rows(recording.eye_state_rows, window_start, window_end)
            window_imu = _slice_inclusive(recording.imu_samples, window_start, window_end, "timestamp_ns")
            window_blinks = _slice_inclusive(recording.blinks, window_start, window_end, "start_timestamp_ns")
            cog_state, cog_conf, cog_features = cognitive_adapter.predict_from_window(
                window_eye_rows,
                window_imu,
                window_blinks,
            )
        else:
            cog_state = "neutral"
            cog_conf = {"focused": 0.0, "confused": 0.0, "stressed": 0.0, "neutral": 1.0}
            cog_features = None

        word_summary = word_tracker.get_reading_summary() if word_tracker is not None else None
        verdict = fusion_agent.fuse(
            phishing_probability=phish_result.phishing_probability,
            suspicious_keywords=phish_result.suspicious_keywords_found,
            gaze_regions_seen=gaze_state.regions_seen,
            fixation_times=gaze_state.region_dwell_times,
            cognitive_state=cog_state,
            cognitive_confidence=cog_conf,
            words_read=word_summary.words_read if word_summary else None,
            suspicious_words_gazed=word_summary.suspicious_words_gazed if word_summary else None,
            unread_suspicious_words=word_summary.unread_suspicious_words if word_summary else None,
            reading_coverage=word_summary.reading_coverage if word_summary else -1.0,
        )
        verdicts.append(verdict)

        elapsed_s = (window_end - start_ns) / 1_000_000_000.0
        print(f"\n{'-' * 60}")
        print(f"  [NEON] t = {elapsed_s:.1f}s | window = {config.FUSION_INTERVAL_S:.1f}s")
        print(f"  [GAZE] Regions seen  : {gaze_state.regions_seen}")
        print(f"  [COG]  Cognitive state: {cog_state} {cog_conf}")
        if cog_features is not None:
            print(
                "  [COG]  Features      : "
                f"openness={cog_features.avg_eye_openness:.3f}, blink_rate={cog_features.blink_rate:.3f}, "
                f"motion_std={cog_features.head_motion_std:.3f}"
            )
        print(f"  [PHISH] Phishing prob : {phish_result.phishing_probability:.4f}")
        if word_summary is not None:
            print(
                f"  [READ] Words read    : {word_summary.total_words_read}/{word_summary.total_words_in_email} "
                f"({word_summary.reading_coverage:.0%} coverage)"
            )
        print(f"\n{verdict.message}")
        print(f"{'-' * 60}\n")

        if window_end >= end_ns:
            break
        window_start = window_end
        window_end = min(window_start + window_ns, end_ns)

    return NeonSessionResult(verdicts=verdicts, recording=recording)
