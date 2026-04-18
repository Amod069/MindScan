"""
MindScan — Flask Backend
NCI H9DAI Research Project 2026

Loads all 12 models (3 classical + XLM-RoBERTa per dataset)
and serves predictions via a single /predict endpoint.

Run:  python app.py
Open: http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template
import os, time

# Import our prediction module
from predict import load_all_models, predict_all, models_loaded

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────
# Load models once at startup — not per request
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  MindScan — Starting up")
print("="*55)
print("  Loading models... (XLM-RoBERTa takes ~30s on CPU)")

start = time.time()
load_all_models()
elapsed = time.time() - start

print(f"  ✅ All models loaded in {elapsed:.1f}s")
print(f"  🌐 Open: http://localhost:5000")
print("="*55 + "\n")


# ─────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the main UI."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    POST /predict
    Body: { "text": "your text here" }
    Returns: full prediction JSON from all 12 models
    """
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({'error': 'Missing "text" field in request body'}), 400

    text = data['text'].strip()

    if not text:
        return jsonify({'error': 'Text cannot be empty'}), 400

    if len(text) > 5000:
        return jsonify({'error': 'Text too long (max 5000 characters)'}), 400

    if not models_loaded():
        return jsonify({'error': 'Models not loaded yet — try again in a moment'}), 503

    try:
        t0 = time.time()
        result = predict_all(text)
        result['processing_time_ms'] = round((time.time() - t0) * 1000)
        return jsonify(result)
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/health')
def health():
    """Quick health check endpoint."""
    return jsonify({
        'status': 'ok',
        'models_ready': models_loaded()
    })


# ─────────────────────────────────────────────────────────────────
# START
# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
