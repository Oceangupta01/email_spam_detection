"""
Prediction module for the Email/SMS Spam Detection system.

Artifacts are loaded once. If they are missing or were produced by a
different scikit-learn / model version, the model is retrained
automatically so a deployment never runs an incompatible pickle.
"""

import json
import logging
import os
import pickle
from functools import lru_cache

import numpy as np
import sklearn

import train
from data_preprocessing import extract_email_text, transform_text

logger = logging.getLogger(__name__)

MAX_INPUT_LENGTH = 10_000
DEFAULT_THRESHOLD = 0.5


def _artifacts_current() -> bool:
    if not all(os.path.exists(p) for p in (train.MODEL_PATH, train.VECTORIZER_PATH, train.METRICS_PATH)):
        return False
    with open(train.METRICS_PATH) as f:
        meta = json.load(f)
    return meta.get("model_version") == train.MODEL_VERSION and meta.get("sklearn_version") == sklearn.__version__


@lru_cache(maxsize=1)
def load_artifacts():
    """Return (model, vectorizer, metrics, feature_weights)."""
    if _artifacts_current():
        with open(train.MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(train.VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        with open(train.METRICS_PATH) as f:
            metrics = json.load(f)
    else:
        logger.info("Artifacts missing or built with another scikit-learn version – retraining.")
        model, vectorizer, metrics = train.train_pipeline(save=True)
    return model, vectorizer, metrics, train.feature_weights(model)


def spam_probabilities(texts) -> np.ndarray:
    """Spam probability for each text (no validation; texts are truncated to MAX_INPUT_LENGTH)."""
    model, vectorizer, _, _ = load_artifacts()
    cleaned = [transform_text(extract_email_text(str(t)[:MAX_INPUT_LENGTH])) for t in texts]
    return model.predict_proba(vectorizer.transform(cleaned))[:, 1]


def predict(text: str, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Classify a single text as *spam* or *ham*.

    Returns ``{"label": "spam" | "ham", "confidence": float, "spam_probability": float}``.
    Raises ``ValueError`` if *text* is empty or longer than ``MAX_INPUT_LENGTH``.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text must not be empty.")
    if len(text) > MAX_INPUT_LENGTH:
        raise ValueError(f"Input text exceeds maximum length of {MAX_INPUT_LENGTH} characters.")

    prob = float(spam_probabilities([text])[0])
    label = "spam" if prob >= threshold else "ham"
    return {"label": label, "confidence": prob if label == "spam" else 1 - prob, "spam_probability": prob}
