"""
nlp_module/deberta_classifier.py – Phishing email classification using
HuggingFace DeBERTa-v3.

Model: microsoft/deberta-v3-small  (or deberta-v3-base)

Pipeline:
    1. Tokenise email text with the model's tokeniser.
    2. Run a forward pass through DebertaV2ForSequenceClassification.
    3. Return:
       • phishing probability (sigmoid of logit)
       • per-token attention scores (from the last transformer layer)
       • list of tokens with the highest attention (suspicious cues)

The model defaults to a *zero-shot* mode when no fine-tuned weights are
available: it loads the pretrained LM head and treats label-0 = "ham",
label-1 = "phishing".  Fine-tuning on a phishing dataset is recommended
for production accuracy.

A ``fine_tune()`` helper is provided for convenience.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import config
from utils.logging_utils import get_logger

logger = get_logger(__name__)

# ── Lazy imports (heavy) ──────────────────────
_transformers = None  # will be imported on first use


def _import_transformers():
    global _transformers
    if _transformers is None:
        import transformers  # type: ignore
        _transformers = transformers
    return _transformers


# ──────────────────────────────────────────────
# Output schema
# ──────────────────────────────────────────────
@dataclass
class PhishingResult:
    """Output of a single email classification."""
    phishing_probability: float
    label: str                              # "phishing" or "ham"
    top_attention_tokens: List[Tuple[str, float]] = field(default_factory=list)
    suspicious_keywords_found: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "phishing_probability": self.phishing_probability,
            "label": self.label,
            "top_attention_tokens": self.top_attention_tokens,
            "suspicious_keywords_found": self.suspicious_keywords_found,
        }


# ──────────────────────────────────────────────
# Classifier
# ──────────────────────────────────────────────
class DeBERTaPhishingClassifier:
    """Sequence classifier wrapping HuggingFace DeBERTa-v3.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    device : torch.device | str
        Compute device.
    max_length : int
        Maximum token length for truncation.
    """

    LABEL_MAP = {0: "ham", 1: "phishing"}

    def __init__(
        self,
        model_name: str = config.NLP_MODEL_NAME,
        device: torch.device | str = config.DEVICE,
        max_length: int = config.NLP_MAX_SEQ_LEN,
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(device) if isinstance(device, str) else device
        self.max_length = max_length

        if config.MOCK_MODE:
            self.tokenizer = None
            self.model = None
            logger.info("DeBERTa classifier in MOCK mode")
            return

        transformers = _import_transformers()

        # Priority: fine-tuned local → base local → HuggingFace hub
        if os.path.isdir(config.NLP_FINETUNED_DIR):
            load_path = config.NLP_FINETUNED_DIR
            logger.info("Loading FINE-TUNED model from %s", load_path)
        elif os.path.isdir(config.NLP_LOCAL_DIR):
            load_path = config.NLP_LOCAL_DIR
            logger.info("Loading base model from local dir %s", load_path)
        else:
            load_path = model_name   # fallback to HuggingFace hub id
            logger.info("Local model not found — downloading %s from HuggingFace", load_path)

        logger.info("Loading tokenizer from %s …", load_path)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(load_path)

        logger.info("Loading model from %s …", load_path)
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
            load_path,
            num_labels=2,
            output_attentions=True,
            ignore_mismatched_sizes=True,
        ).to(self.device)
        self.model.eval()
        logger.info("DeBERTa model ready on %s", self.device)

    # ── Classify ─────────────────────────────────
    @torch.no_grad()
    def classify(self, email_text: str) -> PhishingResult:
        """Classify a single email and return structured result."""
        if self.model is None or self.tokenizer is None:
            return self._mock_classify(email_text)

        # Tokenise
        enc = self.tokenizer(
            email_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Phishing probability
        logits = outputs.logits.squeeze(0)       # (2,)
        probs = torch.softmax(logits, dim=-1)
        phish_prob = float(probs[1].item())
        label = "phishing" if phish_prob >= config.NLP_PHISHING_THRESHOLD else "ham"

        # Attention-based token importance
        top_tokens = self._extract_attention_tokens(
            outputs.attentions, input_ids, attention_mask
        )

        # Keyword scan
        found_kw = self._scan_keywords(email_text)

        return PhishingResult(
            phishing_probability=round(phish_prob, 4),
            label=label,
            top_attention_tokens=top_tokens,
            suspicious_keywords_found=found_kw,
        )

    # ── Attention extraction ─────────────────────
    def _extract_attention_tokens(
        self,
        attentions: Tuple[torch.Tensor, ...],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Average attention from the last layer across all heads,
        and return the top-k tokens with their attention scores."""
        if not attentions:
            return []

        # attentions[-1]: (batch, heads, seq, seq)
        last_attn = attentions[-1].squeeze(0).mean(dim=0)  # (seq, seq)
        # Column-wise sum → how much each token is attended *to*
        token_importance = last_attn.sum(dim=0)             # (seq,)

        # Mask padding
        mask = attention_mask.squeeze(0).float()
        token_importance = token_importance * mask

        # Decode tokens
        ids = input_ids.squeeze(0).cpu().tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(ids)  # type: ignore[union-attr]

        scored: List[Tuple[str, float]] = []
        for tok, score in zip(tokens, token_importance.cpu().tolist()):
            if tok in ("[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"):
                continue
            scored.append((tok, round(score, 4)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ── Keyword scan ─────────────────────────────
    @staticmethod
    def _scan_keywords(text: str) -> List[str]:
        lower = text.lower()
        return [kw for kw in config.SUSPICIOUS_KEYWORDS if kw in lower]

    # ── Mock classify ────────────────────────────
    @staticmethod
    def _mock_classify(email_text: str) -> PhishingResult:
        """Keyword-only heuristic for mock mode."""
        lower = email_text.lower()
        hits = [kw for kw in config.SUSPICIOUS_KEYWORDS if kw in lower]
        prob = min(len(hits) * 0.15, 0.99)
        return PhishingResult(
            phishing_probability=round(prob, 4),
            label="phishing" if prob >= config.NLP_PHISHING_THRESHOLD else "ham",
            top_attention_tokens=[],
            suspicious_keywords_found=hits,
        )

    # ── Fine-tuning helper ───────────────────────
    def fine_tune(
        self,
        train_texts: List[str],
        train_labels: List[int],
        epochs: int = 3,
        batch_size: int = 8,
        lr: float = 2e-5,
        save_dir: Optional[str] = None,
    ) -> None:
        """Fine-tune the model on (text, label) pairs.

        Parameters
        ----------
        train_texts : list[str]
            Email bodies.
        train_labels : list[int]
            0 = ham, 1 = phishing.
        """
        if self.model is None or self.tokenizer is None:
            logger.error("Cannot fine-tune in mock mode")
            return

        transformers = _import_transformers()
        from torch.utils.data import DataLoader, TensorDataset

        logger.info("Tokenising %d samples …", len(train_texts))
        enc = self.tokenizer(
            train_texts,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        labels = torch.tensor(train_labels, dtype=torch.long)
        dataset = TensorDataset(enc["input_ids"], enc["attention_mask"], labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for ids, mask, lbl in loader:
                ids, mask, lbl = ids.to(self.device), mask.to(self.device), lbl.to(self.device)
                optimizer.zero_grad()
                out = self.model(input_ids=ids, attention_mask=mask)
                loss = loss_fn(out.logits, lbl)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            logger.info("Epoch %d/%d  loss=%.4f", epoch, epochs, total_loss / len(loader))

        self.model.eval()
        sd = save_dir or config.NLP_FINETUNED_DIR
        os.makedirs(sd, exist_ok=True)
        self.model.save_pretrained(sd)
        self.tokenizer.save_pretrained(sd)
        logger.info("Fine-tuned model saved to %s", sd)
