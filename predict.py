"""
Prediction module for the Email/SMS Spam Detection system.

The model and vectoriser are loaded **once** when this module is first
imported.  Subsequent calls to ``predict()`` reuse them.

v2: Returns extended result dict with heuristic signals, threat labels,
and individual probability breakdown for frontend explainability.
"""

import logging
import os
import pickle

import numpy as np

from data_preprocessing import transform_text
from heuristics import compute_heuristic_score, fuse_scores

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

# ---------------------------------------------------------------------------
# Load model and vectoriser once
# ---------------------------------------------------------------------------

def _load_pickle(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found: {path}. "
            "Please run 'python train.py' first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


_model = _load_pickle(MODEL_PATH)
_vectorizer = _load_pickle(VECTORIZER_PATH)

logger.info("Model and vectorizer loaded successfully.")

# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

_LABEL_MAP = {0: "ham", 1: "spam"}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

MAX_INPUT_LENGTH = 10_000  # characters


def predict(text: str) -> dict:
    """Classify a single text as *spam* or *ham* using a hybrid ML + heuristic engine.

    Parameters
    ----------
    text : str
        Raw email / SMS body.

    Returns
    -------
    dict
        {
            "label"            : "spam" | "ham",
            "confidence"       : float,          # final combined confidence
            "ml_spam_prob"     : float,          # raw ML spam probability
            "heuristic_score"  : float,          # rule-based risk score
            "threat_signals"   : list[str],      # triggered threat labels
            "spam_prob"        : float,          # final combined spam probability
        }

    Raises
    ------
    ValueError
        If *text* is empty or exceeds ``MAX_INPUT_LENGTH``.
    """
    if not text or not text.strip():
        raise ValueError("Input text must not be empty.")

    if len(text) > MAX_INPUT_LENGTH:
        raise ValueError(
            f"Input text exceeds maximum length of {MAX_INPUT_LENGTH} characters."
        )

    # 1. Heuristic score (on raw text to preserve structural signals)
    heuristic_score, threat_signals = compute_heuristic_score(text)

    # 2. Preprocess (enhanced pipeline with placeholder tokens)
    cleaned = transform_text(text)

    # 3. Vectorise
    vector = _vectorizer.transform([cleaned])

    # 4. ML prediction
    prediction = _model.predict(vector)[0]

    # 5. ML spam probability
    if hasattr(_model, "predict_proba"):
        probabilities = _model.predict_proba(vector)[0]
        # probabilities[1] = spam probability
        ml_spam_prob = float(probabilities[1])
    else:
        ml_spam_prob = 1.0 if prediction == 1 else 0.0

    # 6. Fuse ML + heuristic into final spam probability
    final_spam_prob = fuse_scores(ml_spam_prob, heuristic_score)

    # 7. Final label & confidence
    final_label = "spam" if final_spam_prob >= 0.50 else "ham"
    confidence = final_spam_prob if final_label == "spam" else (1.0 - final_spam_prob)

    return {
        "label": final_label,
        "confidence": round(confidence, 4),
        "ml_spam_prob": round(ml_spam_prob, 4),
        "heuristic_score": round(heuristic_score, 4),
        "threat_signals": threat_signals,
        "spam_prob": round(final_spam_prob, 4),
    }
