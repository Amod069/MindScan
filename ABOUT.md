# MindScan — Complete Project Reference
### NCI H9DAI · MSc Artificial Intelligence · 2026

> **Disclaimer:** This is a research prototype built for academic coursework. It is **not** a clinical tool and must never be used for actual medical diagnosis or mental health assessment.

---

## Table of Contents

1. [Project Purpose](#1-project-purpose)
2. [Tech Stack](#2-tech-stack)
3. [Directory Structure](#3-directory-structure)
4. [Core Features](#4-core-features)
5. [Components & Modules](#5-components--modules)
6. [Data Models & API Contract](#6-data-models--api-contract)
7. [API Endpoints](#7-api-endpoints)
8. [Configuration & Setup](#8-configuration--setup)
9. [AI / ML Architecture](#9-ai--ml-architecture)
10. [Entry Points & Running the App](#10-entry-points--running-the-app)
11. [Dependencies](#11-dependencies)
12. [Testing & Evaluation](#12-testing--evaluation)
13. [Key Findings & Anomalies](#13-key-findings--anomalies)
14. [Research Context](#14-research-context)

---

## 1. Project Purpose

MindScan is a multi-model mental health text analysis system. It runs **12 machine learning classifiers simultaneously** across **3 independent datasets**, returning three clinically distinct assessments from a single text input:

| Assessment | Task | Classes |
|---|---|---|
| Depression Type | Multi-class classification | postpartum, major depressive, bipolar, psychotic, no depression, atypical |
| Binary Depression | Binary classification | Depressed / Not Depressed |
| Suicide Risk | Binary classification | Suicide Risk / No Suicide Risk |

**Research goal:** Extend Tumaliuan et al. (2024) with modern transformer embeddings (XLM-RoBERTa), classical ML gold standards (SVM), and SMOTE balancing — achieving a **+12.7% F1 improvement** (0.81 → 0.9269) over the baseline.

**Key architectural decision — parallel, not sequential:** All 3 datasets run independently. Suicidal ideation can exist without depression markers; a sequential pipeline would gate out those cases entirely. Research Question 4 (RQ4) explicitly tests whether the parallel design catches cases that sequential would miss.

---

## 2. Tech Stack

| Layer | Technology | Version |
|---|---|---|
| Web framework | Flask | 3.0.3 |
| Classical ML | scikit-learn | 1.6.1 |
| Gradient boosting | XGBoost | 2.0.3 |
| Transformer models | HuggingFace Transformers | 4.41.2 |
| Deep learning runtime | PyTorch | 2.3.0 |
| Model serialization | joblib | 1.4.2 |
| Numerical ops | NumPy | 1.26.4 |
| Frontend | HTML5 + vanilla JS + CSS3 | — |
| Fonts | Instrument Serif, Geist, DM Mono | — |

No database. No frontend framework. All model state is held in memory after startup.

---

## 3. Directory Structure

```
MindScan/
├── app.py                              # Flask entry point (94 lines)
├── predict.py                          # Core prediction logic (303 lines)
├── requirements.txt                    # Python dependencies (7 packages)
├── README.md                           # Quick-start guide
├── ABOUT.md                            # This file — full project reference
├── .gitignore                          # models/ excluded (too large for git)
│
├── templates/
│   └── index.html                      # Single-page web UI
│
├── notebooks/
│   ├── DA_Notebook_One.ipynb           # Classical model training (2,269 lines)
│   └── DA_2_Notebook.ipynb             # XLM-RoBERTa training (13,178 lines)
│
├── models/
│   ├── classical/
│   │   ├── le_d1.pkl                   # LabelEncoder — D1 (543 bytes)
│   │   ├── le_d2.pkl                   # LabelEncoder — D2
│   │   ├── le_d3.pkl                   # LabelEncoder — D3
│   │   ├── tfidf_d1.pkl                # TF-IDF vectorizer — D1 (1.4 MB, 34,615 features)
│   │   ├── tfidf_d2.pkl                # TF-IDF vectorizer — D2 (569 KB, 50,000 features)
│   │   ├── tfidf_d3.pkl                # TF-IDF vectorizer — D3 (2.3 MB, 60,000 features)
│   │   ├── logistic_regression_d1.pkl  # LR — D1 (1.6 MB)
│   │   ├── logistic_regression_d2.pkl  # LR — D2 (120 KB)
│   │   ├── logistic_regression_d3.pkl  # LR — D3 (470 KB)
│   │   ├── svm_d1.pkl                  # LinearSVC — D1 (1.6 MB)
│   │   ├── svm_d2.pkl                  # LinearSVC — D2 (120 KB)
│   │   ├── svm_d3.pkl                  # LinearSVC — D3 (470 KB)
│   │   ├── xgboost_d1.pkl              # XGBoost — D1 (3.1 MB)
│   │   ├── xgboost_d2.pkl              # XGBoost — D2 (362 KB)
│   │   ├── xgboost_d3.pkl              # XGBoost — D3 (702 KB)
│   │   ├── random_forest_d1/d2/d3.pkl  # RF — NOT deployed (241 MB + 72 MB + 334 MB)
│   │   ├── classical_results.csv       # Performance metrics table
│   │   └── *.png                       # Confusion matrices + EDA plots (16 images)
│   │
│   └── transformers/
│       ├── xlmr_d1_final/              # Fine-tuned XLM-RoBERTa — D1 (1.1 GB)
│       │   ├── config.json             # Model architecture config
│       │   ├── model.safetensors       # Weights (1.1 GB)
│       │   ├── tokenizer.json          # BPE tokenizer (17 MB)
│       │   └── tokenizer_config.json   # Tokenizer metadata
│       ├── xlmr_d2_final/              # Fine-tuned XLM-RoBERTa — D2 (1.1 GB)
│       └── xlmr_d3_final/              # Fine-tuned XLM-RoBERTa — D3 (1.1 GB)
│
├── venv/                               # Python virtual environment
└── .venv/                              # Backup venv (both in .gitignore)
```

**Total disk usage:** ~3.2 GB (dominated by 3 × 1.1 GB transformer weights)

---

## 4. Core Features

### Four Models Per Dataset (12 Total)

Each of the 3 datasets is evaluated by 4 independent models. All run on every request:

1. Logistic Regression (TF-IDF input)
2. SVM / LinearSVC (TF-IDF input)
3. XGBoost (TF-IDF input)
4. XLM-RoBERTa fine-tuned (raw text input)

### Risk Aggregation

- If **3 or more of the 4 Dataset 3 models** flag suicide risk → `risk_flag = true`
- UI renders a red danger banner
- Response includes `"suicide_votes": "X/4 models flagged suicide risk"`

### Text Preprocessing Pipeline

Applied to all input before TF-IDF vectorization (raw text passed to transformers):

```
lowercase → remove URLs (http/www/https) → strip @mentions
→ remove # symbols (word kept) → delete punctuation → normalize whitespace
```

### UI Features (index.html)

- Live demo textarea with 5000-character limit and real-time counter
- Sample text buttons for quick testing
- Results display: winner card + 4 model confidence bars per dataset
- Class probability breakdown (expandable)
- Risk flag banner (red = danger, green = safe)
- CRISP-DM interactive timeline (6 stages, collapsible detail panels)
- Dataset explorer with class distribution bars
- Model card grid with F1 scores
- Project folder tree with file detail pane
- Animated stat counters in hero section
- Comparison panel vs Tumaliuan et al. (2024) baseline

---

## 5. Components & Modules

### `app.py` (94 lines)

Flask application. Responsibilities:
- Initializes Flask app
- Calls `load_all_models()` at startup (blocks until complete)
- Defines 3 routes: `/`, `/predict`, `/health`
- Input validation: max 5000 chars, non-empty, valid JSON
- Prints startup progress with emoji checkmarks to console
- Serves on `0.0.0.0:5000` (accessible on local network)

### `predict.py` (303 lines)

Core prediction engine. Key functions:

| Function | Purpose |
|---|---|
| `load_all_models()` | Loads all 12 models + encoders + tokenizer into `_models` global dict |
| `clean_text(text)` | Regex-based text cleaning (same logic used in both training notebooks) |
| `predict_classical(text, ds)` | TF-IDF vectorization + sklearn predict / decision_function |
| `predict_transformer(text, ds)` | Tokenization → forward pass → softmax probabilities |
| `predict_all(raw_text)` | Main orchestrator: cleans text, runs all 12 models, returns full result dict |

**Confidence normalization:** All models normalize to 0–1:
- LR / XGBoost: `predict_proba()` (native)
- SVM: `softmax(decision_function())` (LinearSVC has no proba by default)
- Transformer: `softmax(logits)`

**D2 label mapping:** Raw integer labels (0, 1) mapped to `"Not Depressed"` / `"Depressed"`. Handles both `str` and `int` types for robustness.

### `templates/index.html`

Single-page application. All UI logic in vanilla JS:
- `fetch('/predict', { method: 'POST', ... })` — AJAX prediction call
- Tab switching, progress bar animation, accordion expand/collapse
- Counter animations for stats in hero section
- No build step, no bundler, no external JS framework

---

## 6. Data Models & API Contract

### Request

```json
POST /predict
Content-Type: application/json

{ "text": "string — max 5000 characters" }
```

### Response

```json
{
  "dataset1": {
    "task": "Depression Type (6 Classes)",
    "models": {
      "Logistic Regression": { "label": "postpartum", "confidence": 0.958 },
      "SVM":                  { "label": "postpartum", "confidence": 0.828 },
      "XGBoost":              { "label": "postpartum", "confidence": 0.999 },
      "XLM-RoBERTa":         { "label": "postpartum", "confidence": 0.997 }
    },
    "winner_model": "XGBoost",
    "winner_prediction": "postpartum",
    "winner_confidence": 0.999,
    "class_probs": {
      "postpartum": 0.997,
      "bipolar": 0.001,
      "major depressive": 0.001,
      "psychotic": 0.0,
      "no depression": 0.0,
      "atypical": 0.001
    }
  },
  "dataset2": {
    "task": "Binary Depression Detection",
    "models": { ... },
    "winner_model": "XLM-RoBERTa",
    "winner_prediction": "Depressed",
    "winner_confidence": 0.998,
    "class_probs": { "Depressed": 0.998, "Not Depressed": 0.002 }
  },
  "dataset3": {
    "task": "Suicide Risk Assessment",
    "models": { ... },
    "winner_model": "XLM-RoBERTa",
    "winner_prediction": "Suicide Risk",
    "winner_confidence": 0.993,
    "class_probs": { "Suicide Risk": 0.993, "No Suicide Risk": 0.007 }
  },
  "risk_flag": true,
  "suicide_votes": "4/4 models flagged suicide risk",
  "processing_time_ms": 2341
}
```

### Internal Model State (`_models` dict in predict.py)

| Key | Type | Description |
|---|---|---|
| `le_d1/d2/d3` | `LabelEncoder` | Decodes integer predictions to class names |
| `tfidf_d1/d2/d3` | `TfidfVectorizer` | Converts cleaned text to sparse feature vectors |
| `logistic_regression_d1/d2/d3` | `LogisticRegression` | Linear baseline |
| `svm_d1/d2/d3` | `LinearSVC` | SVM classifier |
| `xgboost_d1/d2/d3` | `XGBClassifier` | Gradient boosting |
| `tokenizer` | `XLMRobertaTokenizer` | Shared BPE tokenizer (all 3 transformer models) |
| `xlmr_d1/d2/d3` | `XLMRobertaForSequenceClassification` | Fine-tuned transformer |
| `xlmr_d1/d2/d3_len` | `int` | Max token length: 128 / 128 / 256 |
| `device` | `str` | `'cuda'` or `'cpu'` |

---

## 7. API Endpoints

### `GET /`
Returns `index.html`. No parameters.

### `POST /predict`

| Scenario | HTTP Status | Response |
|---|---|---|
| Success | 200 | Full prediction JSON (see above) |
| Missing `text` field | 400 | `{ "error": "..." }` |
| Empty text | 400 | `{ "error": "..." }` |
| Text > 5000 chars | 400 | `{ "error": "..." }` |
| Models not loaded yet | 503 | `{ "error": "..." }` |
| Prediction exception | 500 | `{ "error": "..." }` |

**Typical latency:** ~2–3 seconds on CPU (XLM-RoBERTa dominates inference time)

### `GET /health`

```json
{ "status": "ok", "models_ready": true }
```

Use for polling during the ~30-second startup window.

---

## 8. Configuration & Setup

### Environment Variables
None required. All paths computed relative to `app.py` using `os.path.dirname(__file__)`.

### Setup Steps

```bash
# 1. Download models from Google Drive → place in models/classical/ and models/transformers/

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start server
python app.py

# 5. Open browser
# http://localhost:5000
```

### Expected Startup Console Output

```
=======================================================
  MindScan — Starting up
=======================================================
  Loading models... (XLM-RoBERTa takes ~30s on CPU)
  ✓ Loaded encoders/tfidf for d1
  ✓ Loaded encoders/tfidf for d2
  ✓ Loaded encoders/tfidf for d3
  ✓ Loaded logistic_regression_d1
  ✓ Loaded svm_d1
  ✓ Loaded xgboost_d1
  ... (repeated for d2, d3)
  ✓ Using device: cpu
  ✓ Tokeniser loaded
  ✓ Loaded XLM-RoBERTa d1 (max_length=128)
  ✓ Loaded XLM-RoBERTa d2 (max_length=128)
  ✓ Loaded XLM-RoBERTa d3 (max_length=256)
  ✅ All models ready
  🌐 Open: http://localhost:5000
=======================================================
```

### `.gitignore` Exclusions
- `models/` — All trained model files (too large for git, download separately)
- `venv/`, `.venv/` — Virtual environments
- `__pycache__/`, `*.pyc`, `*.pyo`
- `.ipynb_checkpoints/`
- `.DS_Store`, `Thumbs.db`

---

## 9. AI / ML Architecture

### The Three Datasets

| | D1 | D2 | D3 |
|---|---|---|---|
| **Source** | Nusrat et al. (2024) — Zenodo 14233292 | albertobellardini — Kaggle | nikhileswarkomati — Kaggle |
| **Platform** | Twitter | Twitter | Reddit |
| **Size** | 14,983 tweets | 10,314 tweets | 50,000 posts |
| **Task** | 6-class depression type | Binary depression | Binary suicide risk |
| **Avg text length** | 31.4 words | ~30 words | 62–200 words |
| **Class balance** | Imbalanced (1.89×) | Severely imbalanced (3.46×) | Balanced (1.0×) |
| **SMOTE applied** | Yes — 11,986 → 17,982 | Yes — 8,251 → 12,800 | No |

All datasets use stratified 80/20 train/test split, `random_state=42`. Test sets are never touched by SMOTE (realistic evaluation).

### TF-IDF Feature Dimensions

| Dataset | Features |
|---|---|
| D1 | 34,615 |
| D2 | 50,000 |
| D3 | 60,000 |

### XLM-RoBERTa Architecture

- **Base model:** `xlm-roberta-base`
- **Parameters:** 278 million
- **Layers:** 12 transformer layers, 12 attention heads
- **Pre-training data:** 2.5 TB of text across 100 languages
- **Fine-tuning:** 3 epochs on Google Colab T4 GPU
- **Weight format:** `safetensors` (more secure and efficient than `.bin`)
- **Max token lengths:** D1 = 128, D2 = 128, D3 = 256
  - D3 uses 256 because Reddit posts average 200.8 words for the suicide class (3.2× longer than non-suicidal posts at 62.2 words)
- **Tokenizer:** Shared single instance across all 3 models (not duplicated)

### Why SVM Beats Transformer on D1

XLM-RoBERTa's contextual embeddings require sufficient token sequence length to demonstrate advantage over bag-of-words TF-IDF. D1 tweets average only 31.4 words — too short for context to matter. SVM achieves F1=0.9269 vs XLM-RoBERTa's lower score on D1. The transformer's advantage grows with text length and dominates on D3 (avg 200+ words).

### Why Random Forest Was Excluded from Deployment

| Model | D1 F1 | D3 F1 | Aggregate Size |
|---|---|---|---|
| XGBoost | competitive | competitive | ~4.2 MB |
| Random Forest | worst on D1 | worst on D3 | **647 MB** |

Size penalty not justified by performance. Random Forest `.pkl` files remain in `models/classical/` but are never loaded by `predict.py`.

### Performance Results

| Dataset | Model | Macro F1 | Cohen's κ |
|---|---|---|---|
| D1 — Depression Type | **SVM** | **0.9269** | **0.9072** |
| D1 — Depression Type | XGBoost | ~0.90 | — |
| D1 — Depression Type | XLM-RoBERTa | lower | — |
| D2 — Binary Depression | **XLM-RoBERTa** | **0.9993** | **0.9986** |
| D3 — Suicide Risk | **XLM-RoBERTa** | **0.9810** | **0.9620** |
| Baseline (Tumaliuan 2024) | — | 0.81 | — |

**Improvement over baseline: +12.7%**

---

## 10. Entry Points & Running the App

### Production (Local)

```bash
python app.py
```

Opens on `http://localhost:5000`. Server also accessible on local network via `http://<your-ip>:5000`.

### Training (Notebooks — Google Colab Only)

| Notebook | Purpose | Runtime Required |
|---|---|---|
| `notebooks/DA_Notebook_One.ipynb` | Train LR, SVM, XGBoost on all 3 datasets; generate metrics CSV and confusion matrix PNGs | CPU (Colab free tier) |
| `notebooks/DA_2_Notebook.ipynb` | Fine-tune XLM-RoBERTa on all 3 datasets; run full model comparison | **T4 GPU** (Colab) |

Both notebooks save outputs to Google Drive at `MindScan_Models/`.

---

## 11. Dependencies

```
flask==3.0.3          Web framework + routing
scikit-learn==1.6.1   LR, LinearSVC, TfidfVectorizer, LabelEncoder, SMOTE metrics
xgboost==2.0.3        Gradient boosting classifier
transformers==4.41.2  XLM-RoBERTa model + tokenizer (HuggingFace)
torch==2.3.0          PyTorch runtime (GPU optional — CUDA auto-detected)
joblib==1.4.2         Pickle serialization for large sklearn objects
numpy==1.26.4         Numerical operations, softmax computation
```

No Node.js / npm dependencies. Pure Python backend, vanilla JS frontend (no build step).

---

## 12. Testing & Evaluation

### No Automated Test Suite

The project has no `pytest`, `unittest`, or CI/CD pipeline. Evaluation is:

**Quantitative (offline, notebook):**
- Macro F1 score (primary metric — handles class imbalance)
- Cohen's Kappa (measures agreement beyond chance — reported for D1)
- Accuracy
- Confusion matrices (saved as PNG to `models/classical/`)
- `classical_results.csv` — full metrics table for all classical models

**Visual (EDA plots):**
- `eda_d1.png`, `eda_d2.png`, `eda_d3.png` — class distributions and text length histograms

**Manual (UI):**
- Sample text buttons in the live demo for smoke-testing the prediction pipeline
- All 4 model predictions + confidence bars shown simultaneously

---

## 13. Key Findings & Anomalies

1. **SVM beats XLM-RoBERTa on D1** — Short tweets (31.4 words avg) don't provide enough context for transformer embeddings to outperform TF-IDF bag-of-words. Classical ML is not always inferior to modern deep learning.

2. **D3 text length asymmetry** — Suicide posts (200.8 words avg) are 3.2× longer than non-suicidal posts (62.2 words). This drove the max_length=256 decision for the D3 transformer.

3. **Near-perfect D2 score (F1=0.9993)** — Binary depression on tweets is almost perfectly separable with XLM-RoBERTa, likely due to strong lexical signals in the dataset.

4. **Parallel architecture prevents missed cases** — Sequential gating (e.g., only check suicide if depression detected) would miss suicidal ideation in people who show no depression markers. All 3 tasks always run.

5. **Confidence computation differs by model type** — SVM uses `softmax(decision_function())` because `LinearSVC` lacks native probability calibration. All outputs are normalized to 0–1 for UI consistency.

6. **Transformer weights in safetensors format** — Newer, more secure format vs. PyTorch `.bin`. Resists pickle deserialization attacks.

7. **SMOTE only on training data** — Oversampling is applied only to training splits. Test sets remain unmodified to reflect real-world class distributions.

8. **Random Forest technically present but never loaded** — The `.pkl` files exist in `models/classical/` but `predict.py` has no code path that loads them.

---

## 14. Research Context

**Project:** NCI H9DAI — Data Analytics for Artificial Intelligence  
**Degree:** MSc Artificial Intelligence  
**Year:** 2026  
**Methodology:** CRISP-DM (6 stages: Business Understanding → Data Understanding → Data Preparation → Modelling → Evaluation → Deployment)

**Baseline paper:** Tumaliuan et al. (2024) — depression detection on Filipino Twitter, F1=0.81

**Research Questions:**
- RQ1: Can classical ML (SVM, LR, XGBoost) exceed the 0.81 baseline?
- RQ2: Can XLM-RoBERTa further improve on classical ML?
- RQ3: Does SMOTE balancing improve F1 on imbalanced datasets?
- RQ4: Does the parallel architecture catch cases a sequential pipeline would miss?

**Datasets used (English/multilingual, broader than baseline's Filipino-only scope):**
- D1: Zenodo 14233292 (Nusrat et al.)
- D2: Kaggle — albertobellardini
- D3: Kaggle — nikhileswarkomati

**Key contributions over baseline:**
- Multi-dataset parallel evaluation (vs. single dataset)
- XLM-RoBERTa multilingual transformer (vs. no transformer)
- SMOTE balancing (vs. no balancing strategy)
- Cohen's Kappa reporting (vs. accuracy/F1 only)
- Explainable per-model confidence scores in UI

---

*NCI H9DAI · Data Analytics for Artificial Intelligence · MSc Artificial Intelligence · 2026*
