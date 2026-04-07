"""
fusion_module/multimodal_fusion_agent.py – Combine signals from gaze, face,
and NLP modules to produce a single risk assessment.

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
    ) -> FusionVerdict:
        """Evaluate all signals and return a verdict.

        The logic is layered:

        1. Base risk from NLP phishing score.
        2. Adjustments from gaze behaviour (did the user inspect critical
           regions like URLs and sender?).
        3. Adjustments from cognitive state (stress or confusion may
           indicate susceptibility).
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
        }

        # ── 1. NLP base risk ─────────────────────
        if phishing_probability >= self.phishing_high:
            risk_score += 4.0
            reasons.append(
                f"NLP model reports very high phishing probability ({phishing_probability:.2f})."
            )
        elif phishing_probability >= self.phishing_med:
            risk_score += 2.5
            reasons.append(
                f"NLP model reports elevated phishing probability ({phishing_probability:.2f})."
            )
        elif phishing_probability >= 0.3:
            risk_score += 1.0
            reasons.append(
                f"NLP model reports moderate phishing probability ({phishing_probability:.2f})."
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
                f"Total email fixation time very short ({total_dwell:.1f}s) – "
                "possible hasty decision."
            )

        # ── 3. Cognitive state adjustments ───────
        if cognitive_state == "stressed":
            risk_score += 1.5
            reasons.append("User appears stressed – higher susceptibility to social engineering.")
        elif cognitive_state == "confused":
            risk_score += 1.0
            reasons.append("User appears confused – may not recognise phishing cues.")
        elif cognitive_state == "focused" and phishing_probability < self.phishing_med:
            risk_score -= 0.5  # user is alert and email looks safe → lower risk

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
            header = "✅ LOW RISK – Email appears safe."
        elif level == "MEDIUM":
            header = "⚠️  MEDIUM RISK – Review the email carefully before acting."
        elif level == "HIGH":
            header = "🚨 HIGH RISK – This email is likely a phishing attempt!"
        else:
            header = "🛑 CRITICAL RISK – DO NOT interact with this email!"

        if reasons:
            body = "\n".join(f"  • {r}" for r in reasons)
            return f"{header}\n{body}"
        return header
