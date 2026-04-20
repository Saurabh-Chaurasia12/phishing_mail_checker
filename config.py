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

# Gaze coordinate mapping:
# - "direct_xy": maps normalized gaze_vec(x,y) directly to screen
# - "angular": uses yaw/pitch projection from (x,y,z)
GAZE_MAPPING_MODE = os.environ.get("MTP_GAZE_MAPPING", "direct_xy")
GAZE_X_GAIN = float(os.environ.get("MTP_GAZE_X_GAIN", "1.2"))
GAZE_Y_GAIN = float(os.environ.get("MTP_GAZE_Y_GAIN", "1.2"))
CALIBRATE_GAZE_EVERY_RUN = os.environ.get("MTP_CALIBRATE_GAZE", "1") == "1"
GAZE_CALIBRATION_POINTS = int(os.environ.get("MTP_GAZE_CALIB_POINTS", "5"))
GAZE_CALIBRATION_SECONDS_PER_POINT = float(os.environ.get("MTP_GAZE_CALIB_SECONDS", "1.2"))
GAZE_CALIBRATION_MIN_SAMPLES = int(os.environ.get("MTP_GAZE_CALIB_MIN_SAMPLES", "5"))

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
ENABLE_FACE_MODULE = os.environ.get("MTP_ENABLE_FACE", "1") == "1"
OPENFACE_BINARY = os.environ.get(
    "OPENFACE_BIN",
    r"C:\OpenFace\FeatureExtraction.exe"  # default Windows path
)
OPENFACE_AVAILABLE = os.path.isfile(OPENFACE_BINARY) and not MOCK_MODE

COG_CLASSIFIER_PATH = os.path.join(MODEL_DIR, "cognitive_classifier.pkl")
COG_FEATURE_WINDOW_S = 3.0   # seconds of features to aggregate

# ──────────────────────────────────────────────
# Phishing Analysis Module
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
# Neon Offline Module
# ──────────────────────────────────────────────
NEON_RECORDING_DIR = os.environ.get("MTP_NEON_RECORDING_DIR", "")
NEON_COORDINATE_MODE = os.environ.get("MTP_NEON_COORDINATE_MODE", "identity")
NEON_ASSUME_SCREEN_SPACE = os.environ.get("MTP_NEON_ASSUME_SCREEN_SPACE", "1") == "1"
NEON_COG_FEATURE_WINDOW_S = float(os.environ.get("MTP_NEON_COG_WINDOW_S", "3.0"))
NEON_ENABLE_COGNITIVE = os.environ.get("MTP_NEON_ENABLE_COGNITIVE", "0") == "1"
NEON_FIXATION_MIN_DURATION_S = float(os.environ.get("MTP_NEON_FIXATION_MIN_DURATION_S", "0.02"))

# ──────────────────────────────────────────────
# OCR Module
# ──────────────────────────────────────────────
OCR_BACKEND = os.environ.get("MTP_OCR_BACKEND", "tesseract").lower()
OCR_MODEL_DIR = os.path.join(MODEL_DIR, "easyocr")
OCR_LANGUAGES = ["en"]                        # language codes for EasyOCR
OCR_GPU = torch.cuda.is_available()           # use GPU if available
OCR_CONFIDENCE_THRESHOLD = 0.3                # min confidence to keep a word
TESSERACT_CMD = os.environ.get("MTP_TESSERACT_CMD", "")
TESSERACT_LANG = os.environ.get("MTP_TESSERACT_LANG", "eng")
OCR_WORD_BOXES_CSV = os.environ.get("MTP_OCR_WORD_BOXES_CSV", os.path.join(LOG_DIR, "ocr_word_boxes.csv"))

# Crop OCR to the mail body area before word tracking.
# Format: (left, top, width, height) relative to the source image/screen.
MAIL_CONTENT_REGION = (190,190,1540,840)

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOG_LEVEL = os.environ.get("MTP_LOG_LEVEL", "INFO")
LOG_TO_FILE = True
