"""
config.py – Central configuration for the Multimodal Phishing Detection prototype.

All tuneable hyper-parameters, model paths, and runtime flags live here so
every module imports a single source of truth.
"""

import os
import torch

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "pretrained_models")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────
# Mock mode – set True to run without heavy models
# ──────────────────────────────────────────────
MOCK_MODE = bool(os.environ.get("MTP_MOCK", "0") == "1")

# ──────────────────────────────────────────────
# Video / Webcam
# ──────────────────────────────────────────────
WEBCAM_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS_TARGET = 30

# ──────────────────────────────────────────────
# Gaze Module
# ──────────────────────────────────────────────
L2CSNET_WEIGHTS_PATH = os.path.join(MODEL_DIR, "l2csnet_gaze360.pkl")
L2CSNET_WEIGHTS_URL = os.environ.get("L2CSNET_WEIGHTS_URL", "")
USE_L2CSNET = os.path.isfile(L2CSNET_WEIGHTS_PATH) and not MOCK_MODE

# Screen calibration (pixels) – adjust to your monitor
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

# Fixation detection
FIXATION_RADIUS_PX = 50        # max dispersion for fixation cluster
FIXATION_MIN_DURATION_S = 0.2  # minimum dwell time (seconds)

# Email interface region bounding boxes (x1, y1, x2, y2) in screen px.
# These should be calibrated per email client / layout.
EMAIL_REGIONS = {
    "sender":      (250, 260,  645, 300),
    "subject":     (250, 210, 900, 260),
    "body":        (250, 300, 1820, 700),
    "url":         (250, 300, 1820, 700),
    "attachments": (250, 700, 1800, 875),
}

# ──────────────────────────────────────────────
# Face / Cognitive Module
# ──────────────────────────────────────────────
OPENFACE_BINARY = os.environ.get(
    "OPENFACE_BIN",
    r"C:\OpenFace\FeatureExtraction.exe"  # default Windows path
)
OPENFACE_AVAILABLE = os.path.isfile(OPENFACE_BINARY) and not MOCK_MODE

COG_CLASSIFIER_PATH = os.path.join(MODEL_DIR, "cognitive_classifier.pkl")
COG_FEATURE_WINDOW_S = 3.0   # seconds of features to aggregate

# ──────────────────────────────────────────────
# NLP Module
# ──────────────────────────────────────────────
NLP_MODEL_NAME = "microsoft/deberta-v3-small"
NLP_LOCAL_DIR = os.path.join(MODEL_DIR, "deberta_v3_small")       # base model stored here
NLP_FINETUNED_DIR = os.path.join(MODEL_DIR, "deberta_phishing_finetuned")  # after fine-tuning
NLP_MAX_SEQ_LEN = 512
NLP_PHISHING_THRESHOLD = 0.5

# Suspicious keyword list (seed)
SUSPICIOUS_KEYWORDS = [
    "urgent", "verify", "click here", "suspended", "password",
    "account", "security", "update", "confirm", "immediately",
    "expire", "login", "credentials", "bank", "paypal",
    "apple", "microsoft", "irs", "reward", "winner",
    "invoice", "attachment", "wire transfer", "social security",
]

# ──────────────────────────────────────────────
# Fusion Module
# ──────────────────────────────────────────────
FUSION_INTERVAL_S = 3.0   # run fusion every N seconds

# Risk thresholds
PHISHING_HIGH_THRESHOLD = 0.9
PHISHING_MED_THRESHOLD = 0.6

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOG_LEVEL = os.environ.get("MTP_LOG_LEVEL", "INFO")
LOG_TO_FILE = True
