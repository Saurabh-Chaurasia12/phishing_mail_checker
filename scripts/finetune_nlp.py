#!/usr/bin/env python3
"""
scripts/finetune_nlp.py – Fine-tune DeBERTa-v3 on a phishing email dataset.

This script:
    1. Loads a phishing dataset (CSV with 'text' and 'label' columns,
       or uses a HuggingFace dataset).
    2. Splits into train/val sets.
    3. Fine-tunes microsoft/deberta-v3-small for binary classification.
    4. Evaluates on the validation set.
    5. Saves the fine-tuned model to pretrained_models/deberta_phishing_finetuned/

Usage:
    # With a local CSV (columns: text, label where label=1 is phishing):
    python scripts/finetune_nlp.py --data phishing_dataset.csv --epochs 5

    # With a HuggingFace dataset:
    python scripts/finetune_nlp.py --hf_dataset "ealvaradob/phishing-dataset" --epochs 3

    # Quick test with synthetic data:
    python scripts/finetune_nlp.py --demo
"""

from __future__ import annotations

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

import config
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def load_csv_dataset(path: str):
    """Load a CSV with 'text' and 'label' columns."""
    import pandas as pd
    df = pd.read_csv(path)
    assert "text" in df.columns and "label" in df.columns, \
        "CSV must have 'text' and 'label' columns"
    texts = df["text"].tolist()
    labels = df["label"].astype(int).tolist()
    return texts, labels


def load_hf_dataset(name: str, text_col: str = "text", label_col: str = "label"):
    """Load a dataset from HuggingFace Hub."""
    from datasets import load_dataset
    ds = load_dataset(name)
    split = ds.get("train", ds[list(ds.keys())[0]])
    texts = split[text_col]
    labels = split[label_col]
    return texts, [int(l) for l in labels]


def generate_demo_data():
    """Generate small synthetic dataset for testing the pipeline."""
    phishing_examples = [
        "URGENT: Your account has been compromised. Click here to verify your identity immediately.",
        "Dear customer, we detected unusual activity. Please confirm your password at this link.",
        "Your PayPal account will be suspended unless you update your billing information now.",
        "Congratulations! You've won a $1000 gift card. Claim your reward by clicking below.",
        "Security alert: Someone tried to access your account. Verify your credentials now.",
        "Your bank account has been locked. Click this link to restore access immediately.",
        "IRS Notice: You have an outstanding tax payment. Submit payment to avoid penalties.",
        "Your Apple ID was used to sign in. If this wasn't you, click here to secure your account.",
        "Wire transfer of $5000 pending. Confirm your account details to receive the funds.",
        "Invoice #4829 attached. Please review and confirm payment by end of day.",
    ]
    ham_examples = [
        "Hi team, the meeting has been rescheduled to 3 PM tomorrow. Please update your calendars.",
        "Here are the quarterly results. Revenue is up 12% compared to last quarter.",
        "Reminder: The project deadline is next Friday. Let me know if you need any help.",
        "Thanks for your email. I'll review the document and get back to you by Wednesday.",
        "The conference registration is now open. Early bird pricing ends March 15th.",
        "Please find attached the minutes from today's meeting for your reference.",
        "Great work on the presentation! The client was very impressed with our proposal.",
        "I'll be out of office next week. Please contact Sarah for urgent matters.",
        "The new software update is available. You can download it from the internal portal.",
        "Happy birthday! The team is planning a small celebration in the break room at noon.",
    ]
    texts = phishing_examples + ham_examples
    labels = [1] * len(phishing_examples) + [0] * len(ham_examples)
    return texts, labels


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DeBERTa for phishing detection")
    parser.add_argument("--data", type=str, help="Path to CSV dataset (text, label columns)")
    parser.add_argument("--hf_dataset", type=str, help="HuggingFace dataset name")
    parser.add_argument("--demo", action="store_true", help="Use synthetic demo data")
    parser.add_argument("--model", type=str, default=config.NLP_MODEL_NAME,
                        help="Base model name")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=config.NLP_MAX_SEQ_LEN)
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(config.MODEL_DIR, "deberta_phishing_finetuned"))
    parser.add_argument("--val_split", type=float, default=0.2)
    args = parser.parse_args()

    # ── Load data ────────────────────────────
    if args.demo:
        logger.info("Using synthetic demo data (20 samples)")
        texts, labels = generate_demo_data()
    elif args.data:
        logger.info("Loading CSV dataset from %s", args.data)
        texts, labels = load_csv_dataset(args.data)
    elif args.hf_dataset:
        logger.info("Loading HuggingFace dataset: %s", args.hf_dataset)
        texts, labels = load_hf_dataset(args.hf_dataset)
    else:
        parser.error("Provide --data, --hf_dataset, or --demo")

    logger.info("Dataset: %d samples (%d phishing, %d ham)",
                len(texts), sum(labels), len(labels) - sum(labels))

    # ── Train/val split ──────────────────────
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=args.val_split, random_state=42, stratify=labels
    )
    logger.info("Train: %d, Val: %d", len(train_texts), len(val_texts))

    # ── Load model and tokenizer ─────────────
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from torch.utils.data import DataLoader, TensorDataset

    device = config.DEVICE

    # Use local model dir if available, otherwise fall back to hub name
    if os.path.isdir(config.NLP_LOCAL_DIR):
        load_path = config.NLP_LOCAL_DIR
        logger.info("Loading from local dir: %s", load_path)
    else:
        load_path = args.model
        logger.info("Local dir not found — downloading %s from HuggingFace", load_path)

    tokenizer = AutoTokenizer.from_pretrained(load_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        load_path, num_labels=2, ignore_mismatched_sizes=True
    ).to(device)

    # ── Tokenize ─────────────────────────────
    def tokenize(texts_list):
        enc = tokenizer(
            texts_list, max_length=args.max_len,
            truncation=True, padding="max_length", return_tensors="pt"
        )
        return enc["input_ids"], enc["attention_mask"]

    train_ids, train_mask = tokenize(train_texts)
    val_ids, val_mask = tokenize(val_texts)
    train_labels_t = torch.tensor(train_labels, dtype=torch.long)
    val_labels_t = torch.tensor(val_labels, dtype=torch.long)

    train_ds = TensorDataset(train_ids, train_mask, train_labels_t)
    val_ds = TensorDataset(val_ids, val_mask, val_labels_t)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # ── Train ────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    logger.info("Starting training for %d epochs on %s", args.epochs, device)
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for ids, mask, lbl in train_loader:
            ids, mask, lbl = ids.to(device), mask.to(device), lbl.to(device)
            optimizer.zero_grad()
            out = model(input_ids=ids, attention_mask=mask)
            loss = loss_fn(out.logits, lbl)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ── Validate ─────────────────────────
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for ids, mask, lbl in val_loader:
                ids, mask = ids.to(device), mask.to(device)
                out = model(input_ids=ids, attention_mask=mask)
                preds = torch.argmax(out.logits, dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_true.extend(lbl.tolist())

        val_acc = accuracy_score(all_true, all_preds)
        logger.info("Epoch %d/%d  loss=%.4f  val_acc=%.4f", epoch, args.epochs, avg_loss, val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    # ── Final evaluation ─────────────────────
    print("\n" + "=" * 50)
    print("VALIDATION RESULTS")
    print("=" * 50)
    print(classification_report(all_true, all_preds, target_names=["ham", "phishing"]))

    # ── Save model ───────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Fine-tuned model saved to %s", args.output_dir)
    print(f"\nModel saved to: {args.output_dir}")
    print("To use it, set NLP_MODEL_NAME in config.py to this path.")


if __name__ == "__main__":
    main()
