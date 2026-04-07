#!/usr/bin/env python3
"""
scripts/download_models.py – Download and verify all pretrained model weights.

This script handles downloading models that should be stored locally under
``pretrained_models/``:
    1. L2CS-Net – PyTorch checkpoint (manual or URL-based download)
    2. DeBERTa  – downloaded from HuggingFace and saved locally
    3. Cognitive classifier – trained locally, no download needed

Usage:
    python scripts/download_models.py              # download everything
    python scripts/download_models.py --gaze       # L2CS-Net only
    python scripts/download_models.py --nlp        # pre-download DeBERTa
    python scripts/download_models.py --verify     # check what's available
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from utils.logging_utils import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────
# L2CS-Net
# ──────────────────────────────────────────────
L2CS_DEFAULT_URL = config.L2CSNET_WEIGHTS_URL


def download_l2csnet():
    """Download L2CS-Net pretrained weights (if URL provided) or print manual steps."""
    dest = config.L2CSNET_WEIGHTS_PATH
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    if os.path.isfile(dest):
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        logger.info("L2CS-Net weights already exist at %s (%.1f MB)", dest, size_mb)
        return True

    if not L2CS_DEFAULT_URL:
        print("\n  No L2CS-Net URL configured for auto-download.")
        print("  Please download the L2CS-Net checkpoint manually and place it at:")
        print(f"  {dest}\n")
        print("  Tip: you can set env var L2CSNET_WEIGHTS_URL to enable auto-download.")
        return False

    print(f"\n  Downloading L2CS-Net weights from:\n  {L2CS_DEFAULT_URL}")
    print(f"  Saving to: {dest}\n")

    try:
        urllib.request.urlretrieve(L2CS_DEFAULT_URL, dest, _progress_hook)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        logger.info("L2CS-Net downloaded successfully (%.1f MB)", size_mb)
        return True
    except Exception as exc:
        logger.error("L2CS-Net download failed: %s", exc)
        print(f"\n  ❌ Auto-download failed. Please download manually:")
        print("     1. Find the official/preferred L2CS-Net checkpoint online")
        print("     2. Download the weights file")
        print(f"     3. Place it at: {dest}")
        return False


def _progress_hook(block_num, block_size, total_size):
    """Download progress callback."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded * 100 / total_size, 100)
        mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        print(f"\r  [{pct:5.1f}%] {mb:.1f} / {total_mb:.1f} MB", end="", flush=True)


# ──────────────────────────────────────────────
# DeBERTa (HuggingFace)
# ──────────────────────────────────────────────
def download_deberta():
    """Download DeBERTa model and save it into pretrained_models/deberta_v3_small/."""
    dest = config.NLP_LOCAL_DIR
    os.makedirs(dest, exist_ok=True)

    # Check if already downloaded
    marker = os.path.join(dest, "config.json")
    if os.path.isfile(marker):
        logger.info("DeBERTa already saved at %s", dest)
        print(f"\n  ✅ DeBERTa already at: {dest}")
        return True

    print(f"\n  Downloading {config.NLP_MODEL_NAME} from HuggingFace...")
    print(f"  Saving to: {dest}\n")

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        print("  Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.NLP_MODEL_NAME)
        tokenizer.save_pretrained(dest)

        print("  Downloading model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            config.NLP_MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True
        )
        model.save_pretrained(dest)

        logger.info("DeBERTa saved to %s", dest)
        print(f"\n  ✅ DeBERTa saved at: {dest}")
        print("  It will be loaded from this folder at runtime (no network needed).")
        return True
    except Exception as exc:
        logger.error("DeBERTa download failed: %s", exc)
        return False


# ──────────────────────────────────────────────
# Verify
# ──────────────────────────────────────────────
def verify_models():
    """Check which models are available."""
    print("\n  ═══ Model Status ═══\n")

    # L2CS-Net
    if os.path.isfile(config.L2CSNET_WEIGHTS_PATH):
        size_mb = os.path.getsize(config.L2CSNET_WEIGHTS_PATH) / (1024 * 1024)
        print(f"  ✅ L2CS-Net       : {config.L2CSNET_WEIGHTS_PATH} ({size_mb:.1f} MB)")
    else:
        print(f"  ❌ L2CS-Net       : NOT FOUND at {config.L2CSNET_WEIGHTS_PATH}")
        print("                      Download manually and place at this path")

    # DeBERTa (local folder)
    if os.path.isfile(os.path.join(config.NLP_LOCAL_DIR, "config.json")):
        print(f"  ✅ DeBERTa        : {config.NLP_LOCAL_DIR}")
    else:
        print(f"  ❌ DeBERTa        : NOT FOUND at {config.NLP_LOCAL_DIR}")
        print(f"                      Run: python scripts/download_models.py --nlp")

    # Fine-tuned DeBERTa
    if os.path.isdir(config.NLP_FINETUNED_DIR):
        print(f"  ✅ DeBERTa (fine) : {config.NLP_FINETUNED_DIR}")
    else:
        print(f"  ⬜ DeBERTa (fine) : Not yet fine-tuned")

    # Cognitive classifier
    if os.path.isfile(config.COG_CLASSIFIER_PATH):
        print(f"  ✅ Cognitive clf  : {config.COG_CLASSIFIER_PATH}")
    else:
        print(f"  ⬜ Cognitive clf  : Not yet trained (using rule-based heuristic)")

    # OpenFace
    if os.path.isfile(config.OPENFACE_BINARY):
        print(f"  ✅ OpenFace       : {config.OPENFACE_BINARY}")
    else:
        print(f"  ⬜ OpenFace       : Not installed (using MediaPipe fallback)")

    # Device
    try:
        import torch
        dev = "CUDA " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    except Exception:
        dev = "CPU"
    print(f"\n  🖥  Device         : {dev}")
    print()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Download pretrained model weights")
    parser.add_argument("--gaze", action="store_true", help="Download L2CS-Net only")
    parser.add_argument("--nlp", action="store_true", help="Pre-download DeBERTa only")
    parser.add_argument("--verify", action="store_true", help="Check model availability")
    parser.add_argument("--all", action="store_true", help="Download everything")
    args = parser.parse_args()

    if args.verify:
        verify_models()
        return

    if not any([args.gaze, args.nlp, args.all]):
        # Default: verify
        verify_models()
        return

    if args.gaze or args.all:
        download_l2csnet()

    if args.nlp or args.all:
        download_deberta()

    print()
    verify_models()


if __name__ == "__main__":
    main()
