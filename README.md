# Phishing Mail Checker

This project now runs a focused offline pipeline for phishing-email review:

1. wait a few seconds and capture an email screenshot, or load an email image/text file
2. OCR the email and keep word-level bounding boxes
3. run LLM phishing analysis on the extracted email
4. load gaze samples from `time.xlsx`
5. map each `(timestamp, x, y)` gaze point to an OCR word
6. mark words as read when dwell time crosses the configured threshold
7. compare risky phrases against what the user actually read
8. generate a final warning telling the user which phishing phrases they likely missed

## Project Structure

```text
.
├── main.py
├── config.py
├── email_phishing_detector.py
├── pipeline/
│   ├── email_capture.py
│   ├── gaze_input.py
│   ├── phishing_analysis.py
│   ├── reading_analysis.py
│   ├── warning_agent.py
│   └── word_lookup.py
├── ocr_module/
│   └── screen_email_ocr.py
├── utils/
│   ├── config_loader.py
│   └── logging_utils.py
└── neon_data/
    └── time.xlsx
```

## Run
1. pip install requirements.txt

```bash
python main.py --email-from-screen

OR

python main.py --gaze-xlsx neon_data/time.xlsx
```

Useful options:

```bash
python main.py --email-from-image email.png
python main.py --email emails/real_mail.txt
python main.py --gaze-xlsx neon_data/time.xlsx --min-read-ms 20
```

## Required Inputs

- `GEMINI_API_KEY` in your environment for phishing analysis and the final warning LLM step
- Tesseract OCR installed locally if you use screenshot/image OCR
- an Excel file containing timestamp, x, and y gaze samples

## GitHub Notes

- do not commit `.env`; use `.env.example` as the template
- `logs/` contains runtime outputs and is intentionally ignored except for `.gitkeep`
- `image.png` is treated as a local test asset and is ignored by default
- `neon_data/time.xlsx` can stay as a sample input, while generated variants are ignored

## Output

The pipeline prints:

- phishing label and probability
- suspicious phrases found by the LLM
- phrases the reader likely missed or only partially read
- a final warning message

It also writes a JSON report to `logs/pipeline_report.json` by default.
