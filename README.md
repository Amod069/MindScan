# MindScan — Mental Health Detection System
### NCI H9DAI Research Project 2026 · MSc Artificial Intelligence

A multi-model mental health text analysis system that runs 12 ML classifiers across 3 datasets simultaneously, returning depression type, binary depression likelihood, and suicide risk scores for any input text.

---

## Project Structure

```
MindScan/
├── app.py              Flask backend — start here
├── predict.py          Prediction logic (all 12 models)
├── requirements.txt    Python dependencies
├── README.md           This file
├── templates/
│   └── index.html      UI (served by Flask at localhost:5000)
├── models/
│   ├── classical/      Download from Google Drive (see below)
│   └── transformers/   Download from Google Drive (see below)
└── notebooks/
    ├── DA_Notebook_One.ipynb   Classical model training
    └── DA_2_Notebook.ipynb     XLM-RoBERTa + comparison
```

---

## Setup

### 1. Download model files from Google Drive
Download `MindScan_Models/` from Google Drive and place the contents like this:
```
models/
├── classical/
│   ├── le_d1.pkl, le_d2.pkl, le_d3.pkl
│   ├── tfidf_d1.pkl, tfidf_d2.pkl, tfidf_d3.pkl
│   ├── logistic_regression_d1.pkl, _d2.pkl, _d3.pkl
│   ├── svm_d1.pkl, _d2.pkl, _d3.pkl
│   └── xgboost_d1.pkl, _d2.pkl, _d3.pkl
└── transformers/
    ├── xlmr_d1_final/
    ├── xlmr_d2_final/
    └── xlmr_d3_final/
```

### 2. Create Python environment
```bash
python -m venv venv

# Mac/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the server
```bash
python app.py
```

### 5. Open the UI
```
http://localhost:5000
```

**Note:** First startup takes ~30 seconds while XLM-RoBERTa models load into memory.

---

## The 3 Datasets

| | Dataset | Source | Size | Task |
|---|---|---|---|---|
| D1 | Nusrat et al. (2024) | Zenodo 14233292 | 14,983 tweets | 6-class depression type |
| D2 | albertobellardini | Kaggle | 10,314 tweets | Binary depression |
| D3 | nikhileswarkomati | Kaggle | 50,000 Reddit posts | Binary suicide risk |

## The 4 Models (per dataset = 12 total)

1. **Logistic Regression** — simple linear baseline
2. **SVM (LinearSVC)** — classical NLP gold standard
3. **XGBoost** — gradient boosting
4. **XLM-RoBERTa** — transformer, contextual embeddings

*Note: Random Forest excluded from deployment (646 MB files, worst performer on D1/D3).*

---

## Real Results

| Dataset | Best Model | Macro F1 | Cohen's Kappa |
|---|---|---|---|
| D1 Depression Type | **SVM** | 0.9269 | 0.9072 |
| D2 Binary Depression | **XLM-RoBERTa** | 0.9993 | 0.9986 |
| D3 Suicide Risk | **XLM-RoBERTa** | 0.9810 | 0.9620 |

**Key finding:** SVM outperforms XLM-RoBERTa on 6-class psychiatric classification (D1). All models exceed the Tumaliuan et al. (2024) benchmark of F1=0.81.

---

## API

**POST /predict**
```json
// Request
{ "text": "your text here" }

// Response
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
    "class_probs": { "postpartum": 0.997, "bipolar": 0.001, ... }
  },
  "dataset2": { ... },
  "dataset3": { ... },
  "risk_flag": false,
  "suicide_votes": "0/4 models flagged suicide risk",
  "processing_time_ms": 2341
}
```

**GET /health**
```json
{ "status": "ok", "models_ready": true }
```

---

## Disclaimer

This system is a research prototype built for academic coursework. It is **not** a clinical tool and must never be used for actual medical diagnosis or mental health assessment. All datasets are from publicly available sources for research purposes only.

---

*NCI H9DAI · Data Analytics for Artificial Intelligence · 2026*
