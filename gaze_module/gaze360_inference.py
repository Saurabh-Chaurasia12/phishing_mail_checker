"""
gaze_module/gaze360_inference.py – Gaze estimation using pretrained L2CS-Net.

Reference:
    A. A. Abdelrahman et al., "L2CS-Net: Fine-Grained Gaze Estimation
    in Unconstrained Environments", 2023.

Expected checkpoint location:
    pretrained_models/l2csnet_gaze360.pkl (see config.L2CSNET_WEIGHTS_PATH)

If weights are missing, the estimator runs in mock mode and the runtime can
still use the MediaPipe gaze fallback path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

import config
from utils.logging_utils import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────
# L2CS-Net minimal backbone
# ──────────────────────────────────────────────
class _L2CSBackbone(nn.Module):
    """Simplified L2CS-Net style backbone.

    For this prototype we use a ResNet backbone with a compact gaze head that
    predicts a 3-D gaze direction vector directly.
    """

    def __init__(self) -> None:
        super().__init__()
        import torchvision.models as models

        resnet = models.resnet18(weights=None)
        in_features = resnet.fc.in_features
        resnet.fc = nn.Identity()
        self.backbone = resnet
        self.fc = nn.Linear(in_features, 3)  # 3-D gaze direction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W)."""
        feats = self.backbone(x)
        return self.fc(feats)


# ──────────────────────────────────────────────
# Data class for gaze output
# ──────────────────────────────────────────────
@dataclass
class GazeResult:
    """Compact output of a single gaze inference."""
    gaze_vector: Tuple[float, float, float]   # unit 3-D gaze direction
    screen_x: float                           # mapped screen x (pixels)
    screen_y: float                           # mapped screen y (pixels)
    confidence: float                         # 0-1

    def to_dict(self) -> dict:
        return {
            "gaze_vector": self.gaze_vector,
            "screen_x": self.screen_x,
            "screen_y": self.screen_y,
            "confidence": self.confidence,
        }


# ──────────────────────────────────────────────
# Preprocessing transform
# ──────────────────────────────────────────────
_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


# ──────────────────────────────────────────────
# L2CS-Net Estimator
# ──────────────────────────────────────────────
class L2CSNetEstimator:
    """Load L2CS-Net weights and run inference on a single face crop.

    Parameters
    ----------
    weights_path : str | None
        Path to L2CS-Net checkpoint. If *None* or file missing,
        the estimator works in **mock mode** (returns random gaze).
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
    ) -> None:
        self.device = config.DEVICE
        self._mock = True  # default until weights load

        self.model = _L2CSBackbone().to(self.device)
        self.model.eval()

        wp = weights_path or config.L2CSNET_WEIGHTS_PATH
        if wp and config.USE_L2CSNET:
            try:
                ckpt = torch.load(wp, map_location=self.device)
                state = ckpt.get("state_dict", ckpt)
                # Strip "module." prefix from DataParallel checkpoints
                state = {k.replace("module.", ""): v for k, v in state.items()}
                self.model.load_state_dict(state, strict=False)
                self._mock = False
                logger.info("L2CS-Net weights loaded from %s", wp)
            except Exception as exc:
                logger.warning("Failed to load L2CS-Net weights: %s – using mock mode", exc)
        else:
            logger.info("L2CS-Net weights not found; running in mock mode")

    # ── public API ──────────────────────────────
    @torch.no_grad()
    def estimate(self, face_crop: np.ndarray) -> GazeResult:
        """Run gaze estimation on a BGR face crop.

        Parameters
        ----------
        face_crop : np.ndarray
            Face region cropped from the webcam frame (BGR, uint8).

        Returns
        -------
        GazeResult
        """
        if self._mock:
            return self._mock_estimate()

        # Preprocess
        rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        tensor = _transform(rgb).unsqueeze(0).to(self.device)  # (1, 3, 224, 224)
        gaze_vec = self.model(tensor).squeeze(0).cpu().numpy()  # (3,)

        # Normalise
        norm = np.linalg.norm(gaze_vec) + 1e-8
        gaze_vec = gaze_vec / norm

        # Map gaze vector to screen coordinates.
        if getattr(config, "GAZE_MAPPING_MODE", "direct_xy") == "angular":
            yaw = math.atan2(gaze_vec[0], -gaze_vec[2])
            pitch = math.asin(np.clip(gaze_vec[1], -1, 1))
            sx = config.SCREEN_WIDTH / 2 + yaw * (config.SCREEN_WIDTH / math.pi)
            sy = config.SCREEN_HEIGHT / 2 - pitch * (config.SCREEN_HEIGHT / (math.pi / 2))
        else:
            nx = np.clip(0.5 + 0.5 * config.GAZE_X_GAIN * float(gaze_vec[0]), 0.0, 1.0)
            ny = np.clip(0.5 - 0.5 * config.GAZE_Y_GAIN * float(gaze_vec[1]), 0.0, 1.0)
            sx = nx * config.SCREEN_WIDTH
            sy = ny * config.SCREEN_HEIGHT

        sx = float(np.clip(sx, 0, config.SCREEN_WIDTH))
        sy = float(np.clip(sy, 0, config.SCREEN_HEIGHT))

        confidence = float(1.0 - abs(gaze_vec[2]))  # heuristic

        return GazeResult(
            gaze_vector=tuple(gaze_vec.tolist()),  # type: ignore[arg-type]
            screen_x=sx,
            screen_y=sy,
            confidence=round(confidence, 3),
        )

    # ── mock fallback ───────────────────────────
    @staticmethod
    def _mock_estimate() -> GazeResult:
        sx = float(np.random.uniform(0, config.SCREEN_WIDTH))
        sy = float(np.random.uniform(0, config.SCREEN_HEIGHT))
        return GazeResult(
            gaze_vector=(0.0, 0.0, -1.0),
            screen_x=sx,
            screen_y=sy,
            confidence=0.5,
        )
