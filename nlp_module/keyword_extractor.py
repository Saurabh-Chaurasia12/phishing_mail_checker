"""
nlp_module/keyword_extractor.py – Extract and score suspicious keywords from
an email for interpretability.

Two extraction strategies:

1. **Static keyword list** – fast scan against config.SUSPICIOUS_KEYWORDS.
2. **SHAP-based importance** (optional, heavier) – uses the SHAP library to
   attribute the DeBERTa prediction to individual tokens.

The module can be called standalone or used within the fusion pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import config
from utils.logging_utils import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────
# Output schema
# ──────────────────────────────────────────────
@dataclass
class KeywordReport:
    """Result of keyword extraction."""
    keywords_found: List[str] = field(default_factory=list)
    keyword_positions: Dict[str, List[int]] = field(default_factory=dict)
    shap_tokens: List[Tuple[str, float]] = field(default_factory=list)
    risk_score: float = 0.0  # 0-1 aggregated keyword risk

    def to_dict(self) -> dict:
        return {
            "keywords_found": self.keywords_found,
            "keyword_positions": self.keyword_positions,
            "shap_tokens": self.shap_tokens,
            "risk_score": self.risk_score,
        }


# ──────────────────────────────────────────────
# Static keyword scanner
# ──────────────────────────────────────────────
def scan_keywords(
    text: str,
    keyword_list: Optional[List[str]] = None,
) -> KeywordReport:
    """Scan *text* for suspicious keywords and return a report.

    Parameters
    ----------
    text : str
        Raw email text.
    keyword_list : list[str], optional
        Override default suspicious keyword list.
    """
    kws = keyword_list or config.SUSPICIOUS_KEYWORDS
    lower = text.lower()

    found: List[str] = []
    positions: Dict[str, List[int]] = {}

    for kw in kws:
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        matches = list(pattern.finditer(text))
        if matches:
            found.append(kw)
            positions[kw] = [m.start() for m in matches]

    # Simple risk heuristic: more keywords → higher risk (capped at 1.0)
    risk = min(len(found) * 0.1, 1.0)

    return KeywordReport(
        keywords_found=found,
        keyword_positions=positions,
        risk_score=round(risk, 3),
    )


# ──────────────────────────────────────────────
# SHAP-based token attribution (optional)
# ──────────────────────────────────────────────
def shap_explain(
    email_text: str,
    classifier: Any,       # DeBERTaPhishingClassifier instance
    max_evals: int = 200,
    top_k: int = 15,
) -> List[Tuple[str, float]]:
    """Use SHAP's ``Explainer`` to attribute prediction to tokens.

    Requires the ``shap`` package.  Returns a list of (token, shap_value)
    tuples sorted by absolute importance (descending).

    Parameters
    ----------
    email_text : str
        The email to explain.
    classifier : DeBERTaPhishingClassifier
        A loaded classifier with ``.model`` and ``.tokenizer``.
    max_evals : int
        Budget for the SHAP explainer (more = slower but more accurate).
    top_k : int
        Return only the top-k tokens.
    """
    try:
        import shap  # type: ignore
    except ImportError:
        logger.warning("shap not installed – skipping SHAP explanation")
        return []

    if classifier.model is None or classifier.tokenizer is None:
        return []

    import torch

    tokenizer = classifier.tokenizer
    model = classifier.model
    device = classifier.device

    # SHAP needs a callable that maps list[str] → np.ndarray (probabilities)
    def predict_fn(texts: List[str]) -> np.ndarray:
        probs = []
        for t in texts:
            enc = tokenizer(
                t, max_length=classifier.max_length,
                truncation=True, padding="max_length",
                return_tensors="pt",
            )
            with torch.no_grad():
                out = model(
                    input_ids=enc["input_ids"].to(device),
                    attention_mask=enc["attention_mask"].to(device),
                )
            p = torch.softmax(out.logits, dim=-1).cpu().numpy()
            probs.append(p[0])
        return np.array(probs)

    explainer = shap.Explainer(predict_fn, tokenizer, output_names=["ham", "phishing"])
    shap_values = explainer([email_text], max_evals=max_evals)

    # shap_values.values: (1, tokens, 2) – take phishing class (index 1)
    vals = shap_values.values[0][:, 1]
    tokens = shap_values.data[0]

    scored = sorted(
        zip(tokens, vals.tolist()),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    return [(t, round(v, 4)) for t, v in scored[:top_k]]
