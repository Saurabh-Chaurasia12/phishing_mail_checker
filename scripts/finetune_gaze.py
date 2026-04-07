#!/usr/bin/env python3
"""
scripts/finetune_gaze.py – Fine-tune the L2CS-Net gaze model on custom gaze data.

This prototype uses an L2CS-Net style backbone that maps a face crop to a
3D gaze direction vector.

Fine-tuning scenarios:
    1. **Domain adaptation** – fine-tune on your own screen/webcam setup to
       improve gaze-to-screen mapping accuracy.
    2. **Custom calibration** – collect (face_crop, screen_point) pairs from
       a calibration routine and fine-tune the mapping layers.

Data format:
    A directory of samples, each containing:
        sample_000/
            frame_0.jpg ... frame_6.jpg   (7 temporal frames)
            label.txt                      (one line: "yaw pitch" in radians)

Usage:
    # Calibration data collection (interactive):
    python scripts/finetune_gaze.py --collect --output_dir calibration_data

    # Fine-tune on collected data:
    python scripts/finetune_gaze.py --data calibration_data --epochs 10

    # Quick demo with synthetic data:
    python scripts/finetune_gaze.py --demo
"""

from __future__ import annotations

import argparse
import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

import config
from utils.logging_utils import get_logger
from gaze_module.gaze360_inference import _L2CSBackbone

logger = get_logger(__name__)


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class GazeDataset(Dataset):
    """Load gaze calibration data.

    Each sample directory has frame_0.jpg..frame_6.jpg + label.txt
    """

    def __init__(self, root_dir: str, temporal_window: int = 7):
        self.samples = []
        self.temporal_window = temporal_window

        for name in sorted(os.listdir(root_dir)):
            sample_dir = os.path.join(root_dir, name)
            if not os.path.isdir(sample_dir):
                continue
            label_path = os.path.join(sample_dir, "label.txt")
            if not os.path.isfile(label_path):
                continue
            self.samples.append(sample_dir)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir = self.samples[idx]

        # Load frames
        frames = []
        for i in range(self.temporal_window):
            fp = os.path.join(sample_dir, f"frame_{i}.jpg")
            if os.path.isfile(fp):
                img = cv2.imread(fp)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                # Repeat last frame if not enough
                img = frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8)
            frames.append(_transform(img))

        seq = torch.stack(frames)  # (T, 3, 224, 224)

        # Load label (yaw, pitch in radians)
        with open(os.path.join(sample_dir, "label.txt")) as f:
            parts = f.read().strip().split()
            yaw, pitch = float(parts[0]), float(parts[1])

        # Convert to 3D unit gaze vector
        gaze_x = -math.cos(pitch) * math.sin(yaw)
        gaze_y = -math.sin(pitch)
        gaze_z = -math.cos(pitch) * math.cos(yaw)
        gaze_vec = torch.tensor([gaze_x, gaze_y, gaze_z], dtype=torch.float32)

        return seq, gaze_vec


class SyntheticGazeDataset(Dataset):
    """Synthetic data for testing the training pipeline."""

    def __init__(self, n_samples: int = 100, temporal_window: int = 7):
        self.n = n_samples
        self.tw = temporal_window

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        seq = torch.randn(self.tw, 3, 224, 224)
        yaw = np.random.uniform(-math.pi / 4, math.pi / 4)
        pitch = np.random.uniform(-math.pi / 6, math.pi / 6)
        gaze_x = -math.cos(pitch) * math.sin(yaw)
        gaze_y = -math.sin(pitch)
        gaze_z = -math.cos(pitch) * math.cos(yaw)
        return seq, torch.tensor([gaze_x, gaze_y, gaze_z], dtype=torch.float32)


# ──────────────────────────────────────────────
# Calibration data collection
# ──────────────────────────────────────────────
def collect_calibration_data(output_dir: str, n_points: int = 9, frames_per_point: int = 7):
    """Interactive calibration: display dots on screen, capture face crops.

    The user looks at each dot while the webcam captures frames.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Grid of calibration points (normalised 0-1)
    grid_size = int(math.sqrt(n_points))
    points = []
    for row in range(grid_size):
        for col in range(grid_size):
            x = (col + 0.5) / grid_size
            y = (row + 0.5) / grid_size
            points.append((x, y))

    cap = cv2.VideoCapture(config.WEBCAM_INDEX)
    if not cap.isOpened():
        logger.error("Cannot open webcam for calibration")
        return

    sw, sh = config.SCREEN_WIDTH, config.SCREEN_HEIGHT

    for i, (nx, ny) in enumerate(points):
        px, py = int(nx * sw), int(ny * sh)

        # Convert screen position to yaw/pitch (approximate)
        yaw = math.atan2(px - sw / 2, sw / 2)
        pitch = math.atan2(sh / 2 - py, sh / 2)

        sample_dir = os.path.join(output_dir, f"sample_{i:03d}")
        os.makedirs(sample_dir, exist_ok=True)

        # Show calibration target
        print(f"\n  Look at the dot at screen position ({px}, {py})")
        print(f"  Press SPACE when ready, then hold gaze for capture...")

        # Wait for space
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            display = frame.copy()
            cv2.putText(display, f"Point {i + 1}/{len(points)}: Look at ({px},{py})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Calibration", display)
            key = cv2.waitKey(1)
            if key == ord(" "):
                break
            elif key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                return

        # Capture frames
        import time
        time.sleep(0.3)  # let user settle
        for f_idx in range(frames_per_point):
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(os.path.join(sample_dir, f"frame_{f_idx}.jpg"), frame)
            time.sleep(0.1)

        # Save label
        with open(os.path.join(sample_dir, "label.txt"), "w") as f:
            f.write(f"{yaw:.6f} {pitch:.6f}\n")

        logger.info("Captured sample %d/%d (yaw=%.3f, pitch=%.3f)", i + 1, len(points), yaw, pitch)

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Calibration data saved to %s (%d samples)", output_dir, len(points))


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
def angular_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean angular error in degrees between predicted and target gaze vectors."""
    pred_norm = pred / (pred.norm(dim=-1, keepdim=True) + 1e-8)
    target_norm = target / (target.norm(dim=-1, keepdim=True) + 1e-8)
    cos_sim = (pred_norm * target_norm).sum(dim=-1).clamp(-1, 1)
    return torch.acos(cos_sim).mean() * (180.0 / math.pi)


def train(args):
    device = config.DEVICE

    # Load or create dataset
    if args.demo:
        logger.info("Using synthetic demo data")
        train_ds = SyntheticGazeDataset(n_samples=50)
        val_ds = SyntheticGazeDataset(n_samples=10)
    else:
        full_ds = GazeDataset(args.data)
        n_val = max(1, int(len(full_ds) * 0.2))
        n_train = len(full_ds) - n_val
        train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Load model
    model = _L2CSBackbone().to(device)
    if os.path.isfile(config.L2CSNET_WEIGHTS_PATH):
        ckpt = torch.load(config.L2CSNET_WEIGHTS_PATH, map_location=device)
        state = ckpt.get("state_dict", ckpt)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        logger.info("Loaded pretrained L2CS-Net weights for fine-tuning")
    else:
        logger.warning("No pretrained weights found — training from scratch")

    # Optionally freeze backbone and only train the final gaze head.
    if args.freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen — only training final FC layer")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    loss_fn = nn.MSELoss()

    logger.info("Training for %d epochs on %s", args.epochs, device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for seq, gaze in train_loader:
            seq, gaze = seq.to(device), gaze.to(device)
            optimizer.zero_grad()
            pred = model(seq[:, -1, :, :, :])
            loss = loss_fn(pred, gaze)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validate
        model.eval()
        ang_errors = []
        with torch.no_grad():
            for seq, gaze in val_loader:
                seq, gaze = seq.to(device), gaze.to(device)
                pred = model(seq[:, -1, :, :, :])
                err = angular_error(pred, gaze)
                ang_errors.append(err.item())

        avg_err = np.mean(ang_errors) if ang_errors else 0
        logger.info("Epoch %d/%d  loss=%.4f  val_angular_error=%.2f°",
                     epoch, args.epochs, total_loss / len(train_loader), avg_err)

    # Save
    save_path = args.output or config.L2CSNET_WEIGHTS_PATH.replace(
        ".pkl", "_finetuned.pkl"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, save_path)
    logger.info("Fine-tuned gaze model saved to %s", save_path)
    print(f"\nModel saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune L2CS-Net model")
    sub = parser.add_subparsers(dest="command")

    # Collect calibration data
    collect_p = sub.add_parser("collect", help="Collect calibration data with webcam")
    collect_p.add_argument("--output_dir", type=str, default="calibration_data")
    collect_p.add_argument("--n_points", type=int, default=9)

    # Train
    train_p = sub.add_parser("train", help="Fine-tune L2CS-Net gaze model")
    train_p.add_argument("--data", type=str, help="Path to calibration data directory")
    train_p.add_argument("--demo", action="store_true", help="Use synthetic data")
    train_p.add_argument("--epochs", type=int, default=10)
    train_p.add_argument("--batch_size", type=int, default=4)
    train_p.add_argument("--lr", type=float, default=1e-4)
    train_p.add_argument("--freeze_backbone", action="store_true",
                         help="Freeze ResNet backbone, fine-tune only final FC layer")
    train_p.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    if args.command == "collect":
        collect_calibration_data(args.output_dir, args.n_points)
    elif args.command == "train":
        train(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
