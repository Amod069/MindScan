"""
MindScan — Prediction Logic
NCI H9DAI Research Project 2026

All model loading and prediction functions.
Imported by app.py — do not run directly.

Datasets:
  D1 — Zenodo (Nusrat 2024) — 6-class depression type
  D2 — Kaggle (albertobellardini) — binary depression (labels: '0'/'1')
  D3 — Kaggle (nikhileswarkomati) — binary suicide risk

Models per dataset:
  Logistic Regression, SVM, XGBoost, XLM-RoBERTa
  (Random Forest excluded — 646 MB, worst performer on D1/D3)
"""

import os, re, string, joblib
import numpy as np

# ─────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
CLASSICAL_DIR  = os.path.join(BASE_DIR, 'models', 'classical')
TRANSFORMER_DIR = os.path.join(BASE_DIR, 'models', 'transformers')

# ─────────────────────────────────────────────────────────────────
# D2 LABEL MAPPING
# The dataset uses '0' and '1' as labels.
# We map them to human-readable strings for the UI.
# ─────────────────────────────────────────────────────────────────
D2_LABEL_MAP = {
    '0': 'Not Depressed',
    '1': 'Depressed',
    0: 'Not Depressed',
    1: 'Depressed',
}

# ─────────────────────────────────────────────────────────────────
# MODEL STORAGE — populated by load_all_models()
# ─────────────────────────────────────────────────────────────────
_models = {}
_loaded = False


def models_loaded():
    return _loaded


def load_all_models():
    """
    Loads all 12 models (4 per dataset × 3 datasets) into memory.
    Called once at server startup. Takes ~30s on CPU due to XLM-RoBERTa.
    """
    global _loaded

    # ── Classical support files ───────────────────────────────────
    for ds in ['d1', 'd2', 'd3']:
        _models[f'le_{ds}']    = joblib.load(os.path.join(CLASSICAL_DIR, f'le_{ds}.pkl'))
        _models[f'tfidf_{ds}'] = joblib.load(os.path.join(CLASSICAL_DIR, f'tfidf_{ds}.pkl'))
        print(f"  ✓ Loaded encoders/tfidf for {ds}")

    # ── Classical models ──────────────────────────────────────────
    for model_name in ['logistic_regression', 'svm', 'xgboost']:
        for ds in ['d1', 'd2', 'd3']:
            key  = f'{model_name}_{ds}'
            path = os.path.join(CLASSICAL_DIR, f'{key}.pkl')
            _models[key] = joblib.load(path)
            print(f"  ✓ Loaded {key}")

    # ── XLM-RoBERTa transformers ──────────────────────────────────
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _models['device'] = device
        print(f"  ✓ Using device: {device}")

        # Shared tokenizer (all 3 models use the same base tokeniser)
        tokenizer_path = os.path.join(TRANSFORMER_DIR, 'xlmr_d1_final')
        _models['tokenizer'] = AutoTokenizer.from_pretrained(tokenizer_path)
        print("  ✓ Tokeniser loaded")

        for ds, max_len in [('d1', 128), ('d2', 128), ('d3', 256)]:
            folder = os.path.join(TRANSFORMER_DIR, f'xlmr_{ds}_final')
            model  = AutoModelForSequenceClassification.from_pretrained(folder)
            model  = model.to(device)
            model.eval()
            _models[f'xlmr_{ds}']     = model
            _models[f'xlmr_{ds}_len'] = max_len
            print(f"  ✓ Loaded XLM-RoBERTa {ds} (max_length={max_len})")

    except Exception as e:
        print(f"  ⚠ XLM-RoBERTa failed to load: {e}")
        print("    Classical models will still work.")

    _loaded = True
    print("  ✅ All models ready")


# ─────────────────────────────────────────────────────────────────
# TEXT CLEANING — same function used in both notebooks
# ─────────────────────────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ─────────────────────────────────────────────────────────────────
# PREDICTION HELPERS
# ─────────────────────────────────────────────────────────────────
def predict_classical(text_clean, ds):
    """
    Runs text through the 3 classical models for one dataset.
    Returns dict: { model_name: {label, confidence} }
    """
    tfidf = _models[f'tfidf_{ds}']
    le    = _models[f'le_{ds}']
    vec   = tfidf.transform([text_clean])

    results = {}
    display_names = {
        'logistic_regression': 'Logistic Regression',
        'svm':                 'SVM',
        'xgboost':             'XGBoost',
    }

    for key, display in display_names.items():
        model    = _models[f'{key}_{ds}']
        pred_idx = model.predict(vec)[0]
        raw_label = le.classes_[pred_idx]

        # Map D2 numeric labels to readable strings
        if ds == 'd2':
            label = D2_LABEL_MAP.get(raw_label, str(raw_label))
        else:
            label = str(raw_label)

        # Confidence: predict_proba if available, else softmax of decision_function
        if hasattr(model, 'predict_proba'):
            conf = float(model.predict_proba(vec)[0][pred_idx])
        elif hasattr(model, 'decision_function'):
            scores = model.decision_function(vec)[0]
            if np.ndim(scores) == 0:
                scores = np.array([float(-scores), float(scores)])
            e    = np.exp(scores - scores.max())
            conf = float(e[pred_idx] / e.sum())
        else:
            conf = 1.0

        results[display] = {
            'label':      label,
            'confidence': round(conf, 4),
        }

    return results


def predict_transformer(text_raw, ds):
    """
    Runs text through XLM-RoBERTa for one dataset.
    Returns { label, confidence, all_probs }
    all_probs = { class_name: probability } for all classes.
    Used for the class breakdown bars in the UI.
    """
    if f'xlmr_{ds}' not in _models:
        return None

    import torch

    model   = _models[f'xlmr_{ds}']
    tok     = _models['tokenizer']
    le      = _models[f'le_{ds}']
    max_len = _models[f'xlmr_{ds}_len']
    device  = _models.get('device', 'cpu')

    inputs = tok(
        text_raw,
        return_tensors='pt',
        max_length=max_len,
        truncation=True,
        padding='max_length'
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs    = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(probs.argmax())
    raw_label = le.classes_[pred_idx]

    if ds == 'd2':
        label = D2_LABEL_MAP.get(raw_label, str(raw_label))
    else:
        label = str(raw_label)

    # Build all_probs dict with readable labels
    all_probs = {}
    for i, p in enumerate(probs):
        raw = le.classes_[i]
        readable = D2_LABEL_MAP.get(raw, str(raw)) if ds == 'd2' else str(raw)
        all_probs[readable] = round(float(p), 4)

    return {
        'label':      label,
        'confidence': round(float(probs[pred_idx]), 4),
        'all_probs':  all_probs,
    }


# ─────────────────────────────────────────────────────────────────
# MAIN FUNCTION — called by Flask /predict endpoint
# ─────────────────────────────────────────────────────────────────
def predict_all(raw_text):
    """
    Runs text through all 12 models across 3 datasets.

    Returns dict:
    {
      dataset1: {
        task, models: {LR, SVM, XGBoost, XLM-RoBERTa},
        winner_model, winner_prediction, winner_confidence,
        class_probs   ← only D1, 6-class breakdown from XLM-RoBERTa
      },
      dataset2: { same structure, D2 labels mapped to readable strings },
      dataset3: { same structure },
      risk_flag: bool,   ← True if ≥3 of 4 D3 models say "suicide"
      suicide_votes: "N/4 models flagged suicide risk",
      winner_summary: { depression_type, depressed, suicide_risk }
    }
    """
    clean = clean_text(raw_text)

    # ── Dataset 1: Depression type ────────────────────────────────
    d1 = predict_classical(clean, 'd1')
    xlmr1 = predict_transformer(raw_text, 'd1')
    if xlmr1:
        d1['XLM-RoBERTa'] = {k: xlmr1[k] for k in ('label','confidence')}

    d1_winner = max(d1.items(), key=lambda x: x[1]['confidence'])

    # ── Dataset 2: Binary depression ─────────────────────────────
    d2 = predict_classical(clean, 'd2')
    xlmr2 = predict_transformer(raw_text, 'd2')
    if xlmr2:
        d2['XLM-RoBERTa'] = {k: xlmr2[k] for k in ('label','confidence')}

    d2_winner = max(d2.items(), key=lambda x: x[1]['confidence'])

    # ── Dataset 3: Suicide risk ───────────────────────────────────
    d3 = predict_classical(clean, 'd3')
    xlmr3 = predict_transformer(raw_text, 'd3')
    if xlmr3:
        d3['XLM-RoBERTa'] = {k: xlmr3[k] for k in ('label','confidence')}

    d3_winner = max(d3.items(), key=lambda x: x[1]['confidence'])

    # ── Suicide risk flag — majority vote across 4 D3 models ─────
    suicide_count = sum(
        1 for r in d3.values()
        if 'suicide' in r['label'].lower() and 'non' not in r['label'].lower()
    )
    risk_flag = suicide_count >= 3

    return {
        'dataset1': {
            'task':               'Depression Type (6 Classes)',
            'models':             d1,
            'winner_model':       d1_winner[0],
            'winner_prediction':  d1_winner[1]['label'],
            'winner_confidence':  d1_winner[1]['confidence'],
            'class_probs':        xlmr1.get('all_probs', {}) if xlmr1 else {},
        },
        'dataset2': {
            'task':               'Depressed or Not?',
            'models':             d2,
            'winner_model':       d2_winner[0],
            'winner_prediction':  d2_winner[1]['label'],
            'winner_confidence':  d2_winner[1]['confidence'],
        },
        'dataset3': {
            'task':               'Suicide Risk Detection',
            'models':             d3,
            'winner_model':       d3_winner[0],
            'winner_prediction':  d3_winner[1]['label'],
            'winner_confidence':  d3_winner[1]['confidence'],
        },
        'risk_flag':     risk_flag,
        'suicide_votes': f'{suicide_count}/4 models flagged suicide risk',
        'winner_summary': {
            'depression_type': f"{d1_winner[1]['label']} ({d1_winner[1]['confidence']*100:.1f}% — {d1_winner[0]})",
            'depressed':       f"{d2_winner[1]['label']} ({d2_winner[1]['confidence']*100:.1f}% — {d2_winner[0]})",
            'suicide_risk':    f"{d3_winner[1]['label']} ({d3_winner[1]['confidence']*100:.1f}% — {d3_winner[0]})",
        }
    }
