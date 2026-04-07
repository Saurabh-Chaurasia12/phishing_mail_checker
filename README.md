# Multimodal Phishing Detection – Research Prototype

> **MTP (Master's Thesis Project)**
> Gaze Tracking + Cognitive State Modeling + Transformer-based NLP
> for intelligent phishing email detection.

---

## 1. Overview

This system runs **three pretrained / lightweight AI models in parallel** while a
user reads an email:

| Module   | Model / Tool                                      | Output                                                                                          |
| -------- | ------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **Gaze** | L2CS-Net (PyTorch) / MediaPipe Face Mesh fallback | Screen gaze (x,y), fixation map, email region dwell times                                       |
| **Face** | OpenFace / MediaPipe fallback                     | Action Units, blink rate, head pose → cognitive state (focused / confused / stressed / neutral) |
| **NLP**  | HuggingFace DeBERTa-v3-small                      | Phishing probability, attention-weighted suspicious tokens, keyword scan                        |

A **Fusion Agent** combines all signals into a risk verdict:
`LOW → MEDIUM → HIGH → CRITICAL`.

---

## 2. Project Structure

```
project_root/
├── main.py                          # Entry point & async main loop
├── config.py                        # Central configuration
├── requirements.txt                 # Python dependencies
│
├── gaze_module/
│   ├── gaze360_inference.py         # L2CS-Net PyTorch inference (legacy filename)
│   ├── mediapipe_fallback.py        # MediaPipe iris-based gaze
│   └── gaze_mapper.py              # Fixation detection & region mapping
│
├── face_module/
│   ├── openface_extractor.py        # OpenFace CLI wrapper + MP fallback
│   └── cognitive_classifier.py      # Rule/ML cognitive state classifier
│
├── nlp_module/
│   ├── deberta_classifier.py        # DeBERTa phishing classifier
│   └── keyword_extractor.py         # Keyword scan + SHAP explainability
│
├── fusion_module/
│   └── multimodal_fusion_agent.py   # Rule-based multimodal fusion
│
├── utils/
│   ├── video_stream.py              # Threaded / async webcam capture
│   ├── logging_utils.py             # Centralised logging
│   └── config_loader.py             # YAML config override
│
├── pretrained_models/               # (create manually, add weights here)
└── logs/                            # Auto-created at runtime
```

---

## 3. Installation

### 3.1 Python Environment

```bash
# Create a virtual environment (Python 3.9+)
python -m venv .venv

# Activate
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3.2 PyTorch with GPU support (recommended)

```bash
# CUDA 11.8 example — adjust for your CUDA version:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3.3 Download Pretrained Models

Use the downloader script to verify and fetch the models that can be stored in
`pretrained_models/`:

```bash
python scripts/download_models.py --verify
python scripts/download_models.py --nlp
```

- `--nlp` downloads DeBERTa into `pretrained_models/deberta_v3_small/`.
- `--gaze` checks L2CS-Net and, if you configured a direct URL, downloads it
  into `pretrained_models/l2csnet_gaze360.pkl`.
- `--all` runs both download steps and then prints a status summary.

#### L2CS-Net

1. Download an L2CS-Net checkpoint from the official project/release source you trust.
2. Save or rename it to `l2csnet_gaze360.pkl`.
3. Place it in `pretrained_models/l2csnet_gaze360.pkl`.
4. Optional requirements: `torch`, `torchvision`, `opencv-python`.
5. If you want the script to auto-download it, set `L2CSNET_WEIGHTS_URL` to a direct file URL before running `python scripts/download_models.py --gaze`.

If the file is not present the system automatically falls back to MediaPipe.

#### DeBERTa-v3-small

1. Download it with the script:
  ```bash
  python scripts/download_models.py --nlp
  ```
2. It is stored locally in `pretrained_models/deberta_v3_small/`.
3. Required Python packages: `transformers`, `tokenizers`, `torch`.
4. For fine-tuning, you may also want `datasets` and `accelerate`.

#### OpenFace (optional)

1. Download from [OpenFace releases](https://github.com/TadasBaltrusaitis/OpenFace/releases).
2. Extract and set the environment variable:
  ```bash
  set OPENFACE_BIN=C:\OpenFace\FeatureExtraction.exe    # Windows
  export OPENFACE_BIN=/opt/OpenFace/build/bin/FeatureExtraction  # Linux
  ```
3. If not installed, the system uses a MediaPipe-based fallback automatically.
4. Optional requirement: the external OpenFace executable itself; no Python package is enough.

#### Cognitive classifier

1. No pretrained download is required.
2. The system uses a rule-based heuristic until you train and save a model.
3. To train one, run `python scripts/finetune_cognitive.py train --data <csv>`.
4. Required packages for training: `scikit-learn`, `joblib`.

---

## 4. Running

### Quick start (mock mode – no GPU / webcam / weights needed)

```bash
python main.py --mock --timeout 15
```

### Full mode with webcam

```bash
python main.py --timeout 60
```

### Custom email file

```bash
python main.py --email path/to/email.txt
```

### All arguments

| Argument       | Description                                          |
| -------------- | ---------------------------------------------------- |
| `--email PATH` | Path to a plain-text email file                      |
| `--mock`       | Force mock mode (random gaze/face, keyword-only NLP) |
| `--no-webcam`  | Use synthetic video frames                           |
| `--timeout N`  | Auto-stop after N seconds (0 = manual quit with `q`) |

---

## 5. Configuration

All tuneable parameters live in **`config.py`**.

You can also create a `config_override.yaml` in the project root to patch
values at runtime without editing Python:

```yaml
nlp_model_name: microsoft/deberta-v3-base
screen_width: 2560
screen_height: 1440
fusion_interval_s: 5.0
mock_mode: false
```

---

## 6. Module Details

### 6.1 Gaze Module

- **L2CSNetEstimator**: Loads the L2CS-Net PyTorch checkpoint from
  `pretrained_models/l2csnet_gaze360.pkl`. Produces a
  3-D gaze direction estimate and converts it to screen pixel coordinates via
  yaw/pitch projection.
- **MediaPipeGazeEstimator**: Uses the refined iris landmarks (468–477)
  from MediaPipe Face Mesh. Computes iris displacement relative to eye
  corners → normalised gaze offset → screen mapping.
- **GazeMapper**: Accumulates gaze samples over time, runs I-DT (dispersion-
  threshold) fixation detection, and classifies each fixation into an email
  interface region (sender / subject / body / url / attachments).

### 6.2 Face / Cognitive Module

- **OpenFaceExtractor**: Invokes the FeatureExtraction binary per frame.
  Parses the CSV for Action Units, head pose, and eye features.
- **MediaPipeFaceExtractor**: Fallback that computes Eye Aspect Ratio for
  blink detection, approximate head pose from landmark geometry, and proxy
  Action Units.
- **CognitiveClassifier**: Aggregates frame-level features over a sliding
  time window. Predicts one of four states: _focused, confused, stressed,
  neutral_. Uses a scikit-learn RandomForest if trained; otherwise a
  hand-crafted rule heuristic.

### 6.3 NLP Module

- **DeBERTaPhishingClassifier**: Loads `microsoft/deberta-v3-small` via
  HuggingFace Transformers. Produces phishing probability (softmax),
  attention-weighted token importance, and a suspicious keyword list.
  Includes a `fine_tune()` method for domain adaptation.
- **keyword_extractor**: Static keyword scan + optional SHAP-based
  explainability for model-level token attribution.

### 6.4 Fusion Module

- **FusionAgent**: Rule-based engine that combines all signals:
  - High phishing probability + uninspected URL → CRITICAL
  - High probability + stressed user → HIGH
  - Moderate probability + confused user → MEDIUM
  - Low probability + focused user → LOW
- Produces a `FusionVerdict` with risk level, human-readable message, and
  a full details dict for logging / downstream use.

---

## 7. Example Test Email

A built-in phishing email is included in `main.py` (constant `EXAMPLE_EMAIL`).
It mimics a PayPal suspension scam with:

- Spoofed sender (`paypa1.com`)
- Urgency language
- Suspicious URL
- Fake attachment

Run with `python main.py --mock --timeout 10` to see it classified.

---

## 8. Extending the System

| Task                         | How                                                             |
| ---------------------------- | --------------------------------------------------------------- |
| Replace gaze model           | Implement the same `estimate()` → `GazeResult` interface        |
| Replace face model           | Implement `extract_from_frame()` → `Dict[str, float]`           |
| Replace NLP model            | Implement `classify()` → `PhishingResult`                       |
| Train cognitive classifier   | Collect labelled feature windows, call `train_classifier(X, y)` |
| Fine-tune DeBERTa            | Call `nlp.fine_tune(texts, labels)` with a phishing dataset     |
| Replace fusion rules with ML | Swap `FusionAgent.fuse()` for a learned model                   |

---

## 9. Known Limitations (Research Prototype)

- Gaze-to-screen mapping is uncalibrated (linear projection). A proper
  calibration routine or eye-tracker ground truth is needed.
- The cognitive state classifier ships with heuristic rules. Labelled data
  collection and model training are required for reliable predictions.
- OpenFace per-frame invocation is slow (file I/O). For real-time use, pipe
  frames via its `-device 0` webcam mode or use the MediaPipe fallback.
- The DeBERTa classifier is loaded with random head weights in zero-shot
  mode since no fine-tuned phishing checkpoint is bundled. Fine-tune on a
  phishing dataset (e.g., Nazario phishing corpus or IWSPA) for accuracy.

---

## 10. License

Research use only. Individual model licences (L2CS-Net, OpenFace, DeBERTa)
apply to their respective artefacts.
