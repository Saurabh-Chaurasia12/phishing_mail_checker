"""
fusion_module/multimodal_fusion_agent.py - Combine signals from gaze, face,
and phishing-analysis modules to produce a single risk assessment.

Design:
    The ``FusionAgent`` is deliberately rule-based at this stage so the
    reasoning is interpretable and easy to adjust.  It can later be replaced
    with a learned model (e.g. a small MLP or decision-tree) once labelled
    multimodal data is available.

Input signals (per fusion cycle):
    • phishing_probability  (float 0-1)
    • suspicious_keywords   (list[str])
    • gaze_regions_seen     (list[str])          – email regions fixated
    • fixation_times        (dict[str, float])   – region → dwell seconds
    • cognitive_state       (str)                – focused / confused / stressed / neutral
    • cognitive_confidence  (dict[str, float])   – per-state probabilities

Output:
    FusionVerdict  with risk level, human-readable message, and details dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import config
from utils.logging_utils import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────
# Output schema
# ──────────────────────────────────────────────
RISK_LEVELS = ("LOW", "MEDIUM", "HIGH", "CRITICAL")


@dataclass
class FusionVerdict:
    """Final multimodal risk assessment."""
    risk_level: str                   # LOW / MEDIUM / HIGH / CRITICAL
    message: str                      # human-readable explanation
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "risk_level": self.risk_level,
            "message": self.message,
            "details": self.details,
        }


# ──────────────────────────────────────────────
# Fusion Agent
# ──────────────────────────────────────────────
class FusionAgent:
    """Multimodal fusion engine.

    Parameters
    ----------
    phishing_high : float
        Threshold above which the email is considered very likely phishing.
    phishing_med : float
        Threshold for a medium-confidence phishing signal.
    """

    def __init__(
        self,
        phishing_high: float = config.PHISHING_HIGH_THRESHOLD,
        phishing_med: float = config.PHISHING_MED_THRESHOLD,
    ) -> None:
        self.phishing_high = phishing_high
        self.phishing_med = phishing_med

    def fuse(
        self,
        phishing_probability: float,
        suspicious_keywords: List[str],
        gaze_regions_seen: List[str],
        fixation_times: Dict[str, float],
        cognitive_state: str,
        cognitive_confidence: Optional[Dict[str, float]] = None,
        # ── Word-level gaze data (Level 2) ──
        words_read: Optional[List[str]] = None,
        suspicious_words_gazed: Optional[List[str]] = None,
        unread_suspicious_words: Optional[List[str]] = None,
        reading_coverage: float = -1.0,
    ) -> FusionVerdict:
        """Evaluate all signals and return a verdict.

        The logic is layered:

        1. Base risk from phishing-analysis score.
        2. Adjustments from gaze behaviour (did the user inspect critical
           regions like URLs and sender?).
        3. Adjustments from cognitive state (stress or confusion may
           indicate susceptibility).
        4. Word-level gaze adjustments (did the user read suspicious words?).
        """
        reasons: List[str] = []
        risk_score = 0.0  # internal numeric score for ranking
        details: Dict[str, Any] = {
            "phishing_probability": phishing_probability,
            "suspicious_keywords": suspicious_keywords,
            "gaze_regions_seen": gaze_regions_seen,
            "fixation_times": fixation_times,
            "cognitive_state": cognitive_state,
            "cognitive_confidence": cognitive_confidence or {},
            "words_read": words_read or [],
            "suspicious_words_gazed": suspicious_words_gazed or [],
            "unread_suspicious_words": unread_suspicious_words or [],
            "reading_coverage": reading_coverage,
        }

        # ── 1. Phishing-analysis base risk ─────────────────────
        if phishing_probability >= self.phishing_high:
            risk_score += 4.0
            reasons.append(
                f"Phishing model reports very high phishing probability ({phishing_probability:.2f})."
            )
        elif phishing_probability >= self.phishing_med:
            risk_score += 2.5
            reasons.append(
                f"Phishing model reports elevated phishing probability ({phishing_probability:.2f})."
            )
        elif phishing_probability >= 0.3:
            risk_score += 1.0
            reasons.append(
                f"Phishing model reports moderate phishing probability ({phishing_probability:.2f})."
            )

        # Keyword bonus
        if len(suspicious_keywords) >= 5:
            risk_score += 1.0
            reasons.append(f"{len(suspicious_keywords)} suspicious keywords found.")
        elif len(suspicious_keywords) >= 2:
            risk_score += 0.5

        # ── 2. Gaze adjustments ──────────────────
        url_inspected = "url" in gaze_regions_seen
        sender_inspected = "sender" in gaze_regions_seen

        if phishing_probability >= self.phishing_med and not url_inspected:
            risk_score += 2.0
            reasons.append("Suspicious link present but user has NOT inspected the URL region.")

        if phishing_probability >= self.phishing_med and not sender_inspected:
            risk_score += 1.0
            reasons.append("User has not inspected the sender field.")

        # Short total fixation time
        total_dwell = sum(fixation_times.values())
        if total_dwell < 2.0 and phishing_probability >= self.phishing_med:
            risk_score += 1.0
            reasons.append(
                f"Total email fixation time very short ({total_dwell:.1f}s) - "
                "possible hasty decision."
            )

        # ── 3. Cognitive state adjustments ───────
        if cognitive_state == "stressed":
            risk_score += 1.5
            reasons.append("User appears stressed - higher susceptibility to social engineering.")
        elif cognitive_state == "confused":
            risk_score += 1.0
            reasons.append("User appears confused - may not recognise phishing cues.")
        elif cognitive_state == "focused" and phishing_probability < self.phishing_med:
            risk_score -= 0.5  # user is alert and email looks safe → lower risk

        # ── 4. Word-level gaze adjustments ───────
        if unread_suspicious_words and phishing_probability >= self.phishing_med:
            n_unread = len(unread_suspicious_words)
            risk_score += min(n_unread * 0.8, 2.5)
            preview = ", ".join(unread_suspicious_words[:5])
            reasons.append(
                f"User has NOT read {n_unread} suspicious word(s): [{preview}]"
            )

        if suspicious_words_gazed and phishing_probability >= self.phishing_med:
            reasons.append(
                f"User noticed {len(suspicious_words_gazed)} suspicious word(s)."
            )

        if 0.0 <= reading_coverage < 0.3 and phishing_probability >= self.phishing_med:
            risk_score += 1.0
            reasons.append(
                f"Very low reading coverage ({reading_coverage:.0%}) - "
                "user may be skim-reading a phishing email."
            )
        elif reading_coverage >= 0.7 and phishing_probability < self.phishing_med:
            risk_score -= 0.3  # careful reader on a safe email

        # ── Map numeric score → level ────────────
        if risk_score >= 6.0:
            level = "CRITICAL"
        elif risk_score >= 4.0:
            level = "HIGH"
        elif risk_score >= 2.0:
            level = "MEDIUM"
        else:
            level = "LOW"

        message = self._compose_message(level, reasons)
        details["risk_score_raw"] = round(risk_score, 2)

        verdict = FusionVerdict(risk_level=level, message=message, details=details)
        logger.info("Fusion verdict: %s (raw=%.2f)", level, risk_score)
        return verdict

    # ── Helper ───────────────────────────────────
    @staticmethod
    def _compose_message(level: str, reasons: List[str]) -> str:
        if level == "LOW":
            header = "[OK] LOW RISK - Email appears safe."
        elif level == "MEDIUM":
            header = "[!!] MEDIUM RISK - Review the email carefully before acting."
        elif level == "HIGH":
            header = "[ALERT] HIGH RISK - This email is likely a phishing attempt!"
        else:
            header = "[CRITICAL] CRITICAL RISK - DO NOT interact with this email!"

        if reasons:
            body = "\n".join(f"  * {r}" for r in reasons)
            return f"{header}\n{body}"
        return header
