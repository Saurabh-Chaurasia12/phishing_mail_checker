#!/usr/bin/env python3
"""
scripts/finetune_cognitive.py – Train the cognitive state classifier.

This script:
    1. Loads labelled facial behaviour data (CSV with feature columns + state label).
    2. Trains a RandomForest classifier (or any sklearn model).
    3. Evaluates with cross-validation.
    4. Saves the trained model to pretrained_models/cognitive_classifier.pkl

Data format (CSV):
    avg_eye_openness, blink_rate, head_pitch_std, head_yaw_std,
    head_roll_std, AU04_mean, AU45_mean, state

    Where state ∈ {focused, confused, stressed, neutral}

Usage:
    # With labelled CSV:
    python scripts/finetune_cognitive.py --data cognitive_labels.csv

    # Quick demo with synthetic data:
    python scripts/finetune_cognitive.py --demo

    # Collect data interactively (webcam + self-labelling):
    python scripts/finetune_cognitive.py collect --output_dir cog_data --duration 60
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report

import config
from utils.logging_utils import get_logger
from face_module.cognitive_classifier import (
    COGNITIVE_STATES, CLASSIFIER_FEATURES, FeatureAggregator, train_classifier
)

logger = get_logger(__name__)


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────
def load_csv(path: str):
    """Load CSV with feature columns + 'state' column."""
    import pandas as pd
    df = pd.read_csv(path)
    feature_cols = [c for c in CLASSIFIER_FEATURES if c in df.columns]
    if not feature_cols:
        raise ValueError(f"CSV must contain feature columns: {CLASSIFIER_FEATURES}")
    X = df[feature_cols].values.astype(np.float32)
    y = df["state"].tolist()
    return X, y


def generate_demo_data(n_per_class: int = 50):
    """Generate synthetic labelled feature vectors for each cognitive state."""
    X_list, y_list = [], []

    for state in COGNITIVE_STATES:
        for _ in range(n_per_class):
            if state == "focused":
                vec = [
                    np.random.uniform(0.25, 0.35),   # avg_eye_openness
                    np.random.uniform(0.1, 0.3),      # blink_rate
                    np.random.uniform(0.01, 0.04),    # head_pitch_std (low)
                    np.random.uniform(0.01, 0.04),    # head_yaw_std (low)
                    np.random.uniform(0.01, 0.03),    # head_roll_std (low)
                    np.random.uniform(0.0, 0.8),      # AU04_mean (low brow)
                    np.random.uniform(0.0, 0.5),      # AU45_mean
                ]
            elif state == "confused":
                vec = [
                    np.random.uniform(0.20, 0.35),
                    np.random.uniform(0.2, 0.5),
                    np.random.uniform(0.05, 0.15),
                    np.random.uniform(0.08, 0.20),    # high yaw std (head tilting)
                    np.random.uniform(0.03, 0.10),
                    np.random.uniform(1.5, 3.0),      # high AU04 (furrowed brow)
                    np.random.uniform(0.3, 1.0),
                ]
            elif state == "stressed":
                vec = [
                    np.random.uniform(0.15, 0.30),
                    np.random.uniform(0.5, 1.0),      # high blink rate
                    np.random.uniform(0.04, 0.12),
                    np.random.uniform(0.04, 0.12),
                    np.random.uniform(0.02, 0.08),
                    np.random.uniform(1.0, 2.5),      # tense brow
                    np.random.uniform(0.8, 2.0),      # high AU45
                ]
            else:  # neutral
                vec = [
                    np.random.uniform(0.25, 0.35),
                    np.random.uniform(0.15, 0.35),
                    np.random.uniform(0.02, 0.06),
                    np.random.uniform(0.02, 0.06),
                    np.random.uniform(0.01, 0.04),
                    np.random.uniform(0.3, 1.2),
                    np.random.uniform(0.1, 0.6),
                ]

            X_list.append(vec)
            y_list.append(state)

    return np.array(X_list, dtype=np.float32), y_list


# ──────────────────────────────────────────────
# Interactive data collection
# ──────────────────────────────────────────────
def collect_data(output_dir: str, duration: int = 60, window_s: float = 3.0):
    """Collect labelled cognitive data from webcam.

    The user self-labels their state during recording.
    """
    import cv2
    from face_module.openface_extractor import create_face_extractor

    os.makedirs(output_dir, exist_ok=True)
    extractor = create_face_extractor()
    aggregator = FeatureAggregator(window_s=window_s)

    cap = cv2.VideoCapture(config.WEBCAM_INDEX)
    if not cap.isOpened():
        logger.error("Cannot open webcam")
        return

    samples_X, samples_y = [], []
    current_label = "neutral"

    print("\n  ═══ Cognitive Data Collection ═══")
    print("  Press keys to set your current state:")
    print("    f = focused")
    print("    c = confused")
    print("    s = stressed")
    print("    n = neutral")
    print("    q = quit and save")
    print(f"  Recording for up to {duration} seconds...\n")

    start = time.time()

    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            continue

        # Extract features
        feats = extractor.extract_from_frame(frame)
        aggregator.push(feats)

        # Aggregate every window_s seconds worth of frames
        vec = aggregator.aggregate()
        if vec is not None and len(samples_X) == 0 or \
           (vec is not None and time.time() - start > len(samples_X) * window_s):
            samples_X.append(vec.copy())
            samples_y.append(current_label)
            elapsed = time.time() - start
            print(f"    [{elapsed:6.1f}s] Recorded sample #{len(samples_X)} as '{current_label}'")

        # Display
        cv2.putText(frame, f"State: {current_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Samples: {len(samples_X)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Cognitive Data Collection", frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("f"):
            current_label = "focused"
            print(f"    → Label set to: {current_label}")
        elif key == ord("c"):
            current_label = "confused"
            print(f"    → Label set to: {current_label}")
        elif key == ord("s"):
            current_label = "stressed"
            print(f"    → Label set to: {current_label}")
        elif key == ord("n"):
            current_label = "neutral"
            print(f"    → Label set to: {current_label}")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    if hasattr(extractor, "close"):
        extractor.close()

    # Save as CSV
    if samples_X:
        import pandas as pd
        data = np.array(samples_X)
        df = pd.DataFrame(data, columns=CLASSIFIER_FEATURES)
        df["state"] = samples_y
        csv_path = os.path.join(output_dir, "cognitive_data.csv")
        df.to_csv(csv_path, index=False)
        logger.info("Saved %d samples to %s", len(samples_X), csv_path)
        print(f"\n  Saved {len(samples_X)} samples to {csv_path}")
    else:
        print("  No samples collected.")


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
def train_model(args):
    if args.demo:
        logger.info("Using synthetic demo data")
        X, y = generate_demo_data(n_per_class=args.samples_per_class)
    else:
        logger.info("Loading data from %s", args.data)
        X, y = load_csv(args.data)

    logger.info("Dataset: %d samples, classes: %s",
                len(y), {s: y.count(s) for s in COGNITIVE_STATES})

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Choose classifier
    if args.model == "rf":
        clf = RandomForestClassifier(
            n_estimators=args.n_estimators, random_state=42, class_weight="balanced"
        )
    elif args.model == "gb":
        clf = GradientBoostingClassifier(
            n_estimators=args.n_estimators, random_state=42
        )
    else:
        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")

    # Cross-validation
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy")
    logger.info("5-fold CV accuracy: %.3f ± %.3f", cv_scores.mean(), cv_scores.std())

    # Final training
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print("\n" + "=" * 50)
    print("TEST SET RESULTS")
    print("=" * 50)
    print(classification_report(y_test, y_pred, target_names=COGNITIVE_STATES))

    # Feature importance
    if hasattr(clf, "feature_importances_"):
        print("Feature Importance:")
        for name, imp in sorted(zip(CLASSIFIER_FEATURES, clf.feature_importances_),
                                 key=lambda x: x[1], reverse=True):
            print(f"  {name:25s} {imp:.4f}")

    # Save
    save_path = args.output or config.COG_CLASSIFIER_PATH
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(clf, save_path)
    logger.info("Saved cognitive classifier to %s", save_path)
    print(f"\nModel saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train cognitive state classifier")
    sub = parser.add_subparsers(dest="command")

    # Collect
    collect_p = sub.add_parser("collect", help="Collect labelled data from webcam")
    collect_p.add_argument("--output_dir", type=str, default="cog_data")
    collect_p.add_argument("--duration", type=int, default=60)

    # Train
    train_p = sub.add_parser("train", help="Train classifier")
    train_p.add_argument("--data", type=str, help="Path to labelled CSV")
    train_p.add_argument("--demo", action="store_true", help="Use synthetic data")
    train_p.add_argument("--model", choices=["rf", "gb"], default="rf",
                         help="rf=RandomForest, gb=GradientBoosting")
    train_p.add_argument("--n_estimators", type=int, default=100)
    train_p.add_argument("--samples_per_class", type=int, default=50)
    train_p.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    if args.command == "collect":
        collect_data(args.output_dir, args.duration)
    elif args.command == "train":
        train_model(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
