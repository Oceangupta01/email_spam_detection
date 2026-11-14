"""
Training pipeline for the Email/SMS Spam Detection system.

Usage:
    python train.py

This script:
1. Loads and cleans the SMS Spam Collection dataset (spam.csv).
2. Augments it with modern email spam examples (phishing, 419, crypto, etc.)
3. Applies the enhanced text preprocessing (URL/money/caps placeholders).
4. Vectorises with TF-IDF using bigrams and sublinear TF scaling.
5. Compares Multinomial Naive Bayes, Logistic Regression and
   Linear SVM (calibrated), all with class_weight='balanced'.
6. Selects the best model by F1 score.
7. Saves the best model + vectoriser as pickle files.
"""

import logging
import os
import pickle
import sys

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder

from data_preprocessing import transform_text
from email_spam_data import EMAIL_SPAM_AUGMENTATION

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "spam.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_and_clean_data(path: str) -> pd.DataFrame:
    """Load spam.csv or compatible CSV and return a cleaned DataFrame."""
    logger.info("Loading data from %s", path)
    df = pd.read_csv(path, encoding="latin-1")

    if "v1" in df.columns and "v2" in df.columns:
        df = df[["v1", "v2"]]
        df.columns = ["target", "text"]
    elif "target" not in df.columns or "text" not in df.columns:
        raise ValueError("CSV must contain either (v1, v2) or (target, text) columns.")

    # Encode target: ham=0, spam=1
    encoder = LabelEncoder()
    df["target"] = encoder.fit_transform(df["target"])

    before = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    logger.info("Dropped %d duplicates (%d → %d rows)", before - len(df), before, len(df))

    return df


def augment_with_email_spam(df: pd.DataFrame) -> pd.DataFrame:
    """Merge base SMS dataset with curated email spam augmentation data."""
    aug_rows = []
    for label, text in EMAIL_SPAM_AUGMENTATION:
        aug_rows.append({"target": 1 if label == "spam" else 0, "text": text})
    aug_df = pd.DataFrame(aug_rows)
    combined = pd.concat([df, aug_df], ignore_index=True)
    logger.info(
        "After augmentation: %d total rows (%d spam, %d ham)",
        len(combined),
        (combined["target"] == 1).sum(),
        (combined["target"] == 0).sum(),
    )
    return combined


def preprocess_column(df: pd.DataFrame) -> pd.DataFrame:
    """Apply shared text preprocessing to the *text* column."""
    logger.info("Preprocessing text column …")
    df = df.copy()
    df["transformed_text"] = df["text"].apply(transform_text)
    return df


def build_tfidf(texts, max_features: int = 6000):
    """Fit a TF-IDF vectoriser with bigrams and sublinear scaling, return (vectoriser, matrix)."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),         # unigrams + bigrams for phrase-level spam signals
        sublinear_tf=True,          # log(1 + tf) dampens effect of repeated terms
        min_df=2,                   # ignore terms appearing in only 1 document
    )
    X = vectorizer.fit_transform(texts)
    return vectorizer, X


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train three models with balanced class weights, evaluate and return best."""

    models = {
        "ComplementNB": ComplementNB(),   # handles class imbalance better than MultinomialNB
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=1000, C=1.0, random_state=42
        ),
        "LinearSVC": CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", max_iter=2000, C=0.5, random_state=42)
        ),
    }

    results = []

    for name, model in models.items():
        logger.info("Training %s …", name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec  = recall_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)

        results.append({"name": name, "model": model, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1})
        logger.info("%s  →  Accuracy=%.4f  Precision=%.4f  Recall=%.4f  F1=%.4f", name, acc, prec, rec, f1)
        print(f"\n--- {name} ---")
        print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

    # Pick best by F1
    best = max(results, key=lambda r: r["f1"])
    logger.info("Best model: %s (F1=%.4f)", best["name"], best["f1"])
    return best["model"], best["name"]


def save_artifacts(model, vectorizer, model_path: str, vectorizer_path: str):
    """Persist model and vectoriser as pickle files."""
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Saved model to %s", model_path)

    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    logger.info("Saved vectorizer to %s", vectorizer_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = load_and_clean_data(DATA_PATH)
    df = augment_with_email_spam(df)
    df = preprocess_column(df)

    vectorizer, X = build_tfidf(df["transformed_text"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    best_model, best_name = train_and_evaluate(X_train, X_test, y_train, y_test)
    save_artifacts(best_model, vectorizer, MODEL_PATH, VECTORIZER_PATH)

    print(f"\n✅  Training complete. Best model: {best_name}")
    print(f"    Model saved to  : {MODEL_PATH}")
    print(f"    Vectorizer saved: {VECTORIZER_PATH}")


if __name__ == "__main__":
    main()
