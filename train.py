"""
Training pipeline for the Email/SMS Spam Detection system.

Usage:
    python train.py

1. Loads the SMS Spam Collection (data/spam.csv) and the SpamAssassin email
   corpus (data/emails.csv.gz) and removes duplicates.
2. Applies the shared preprocessing (data_preprocessing.transform_text).
3. Splits 80/20, stratified by source and label, and fits TF-IDF
   (words + word pairs) on the training split only.
4. Trains a class-balanced, probability-calibrated Linear SVM.
5. Benchmarks the original SMS-only pipeline and four alternative models.
6. Saves model.pkl, vectorizer.pkl and metrics.json (dashboard data).
"""

import json
import logging
import os
import pickle
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from data_preprocessing import STOP_WORDS, pretty_term, transform_text

logger = logging.getLogger(__name__)

MODEL_VERSION = 3
RANDOM_STATE = 42
MAX_CHARS = 10_000
SOURCES = ("SMS", "Email")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SMS_PATH = os.path.join(BASE_DIR, "data", "spam.csv")
EMAIL_PATH = os.path.join(BASE_DIR, "data", "emails.csv.gz")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_datasets() -> pd.DataFrame:
    """Return de-duplicated messages with columns text, label (1 = spam), source."""
    sms = pd.read_csv(SMS_PATH, encoding="latin-1")
    # Messages containing commas spill into the unnamed overflow columns.
    sms_text = sms.iloc[:, 1:].fillna("").astype(str).agg(lambda r: ",".join(x for x in r if x), axis=1)
    sms_df = pd.DataFrame({"text": sms_text, "label": (sms.iloc[:, 0] == "spam").astype(int), "source": "SMS"})

    emails = pd.read_csv(EMAIL_PATH).dropna(subset=["text"])
    email_df = pd.DataFrame({"text": emails["text"].astype(str), "label": emails["target"].astype(int), "source": "Email"})

    df = pd.concat([sms_df, email_df], ignore_index=True)
    df["text"] = df["text"].str.slice(0, MAX_CHARS)
    df = df[df["text"].str.strip() != ""]
    return df.drop_duplicates(subset="text").reset_index(drop=True)


def split(df: pd.DataFrame):
    strata = df["source"] + "_" + df["label"].astype(str)
    return train_test_split(df, test_size=0.2, random_state=RANDOM_STATE, stratify=strata)


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000, sublinear_tf=True)


def build_classifier() -> CalibratedClassifierCV:
    return CalibratedClassifierCV(
        LinearSVC(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE), cv=5,
    )


def feature_weights(model) -> np.ndarray:
    """Per-feature spam weight of the calibrated SVM (positive pushes towards spam)."""
    return np.mean([c.estimator.coef_.ravel() for c in model.calibrated_classifiers_], axis=0)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _scores(y_true, y_pred, y_prob=None) -> dict:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "support": {"ham": int((y_true == 0).sum()), "spam": int((y_true == 1).sum())},
    }
    if y_prob is not None:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
        out["average_precision"] = average_precision_score(y_true, y_prob)
    return out


def _by_source(test_df, pred, prob) -> dict:
    y = test_df["label"].to_numpy()
    out = {}
    for s in SOURCES:
        m = (test_df["source"] == s).to_numpy()
        out[s] = _scores(y[m], pred[m], prob[m])
    return out


def _downsample(arrs, n=200):
    idx = np.unique(np.linspace(0, len(arrs[0]) - 1, min(n, len(arrs[0]))).astype(int))
    return [np.round(a[idx], 4).tolist() for a in arrs]


def _baseline(train_df, test_df) -> dict:
    """Reproduce the original pipeline: SMS-only training, signals discarded, 3,000-term vocabulary."""
    def prep(text):
        return " ".join(t for t in transform_text(text).split() if not t.endswith("token"))

    sms_train = train_df[train_df["source"] == "SMS"]
    vec = TfidfVectorizer(max_features=3000)
    X = vec.fit_transform(sms_train["text"].map(prep))
    clf = CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=RANDOM_STATE)).fit(X, sms_train["label"])
    prob = clf.predict_proba(vec.transform(test_df["text"].map(prep)))[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {"overall": _scores(test_df["label"].to_numpy(), pred, prob), "by_source": _by_source(test_df, pred, prob)}


def _compare_models(X_train, y_train, X_test, y_test) -> list:
    models = {
        "Linear SVM (calibrated)": build_classifier(),
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced", C=10),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=5, metric="cosine"),
    }
    rows = []
    for name, model in models.items():
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        prob = model.predict_proba(X_test)[:, 1]
        pred_s = time.perf_counter() - t0
        s = _scores(y_test, (prob >= 0.5).astype(int), prob)
        rows.append({"model": name, "train_seconds": fit_s, "predict_ms_per_msg": 1000 * pred_s / X_test.shape[0], **s})
        logger.info("%-26s F1=%.4f AUC=%.4f", name, s["f1"], s["roc_auc"])
    return rows


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_pipeline(save: bool = True, benchmark: bool = True):
    """Train, evaluate and (optionally) persist artifacts. Returns (model, vectorizer, metrics)."""
    t_start = time.perf_counter()
    df = load_datasets()
    train_df, test_df = split(df)
    logger.info("Loaded %d unique messages (%d train / %d test)", len(df), len(train_df), len(test_df))

    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(train_df["text"].map(transform_text))
    X_test = vectorizer.transform(test_df["text"].map(transform_text))
    y_train, y_test = train_df["label"].to_numpy(), test_df["label"].to_numpy()

    model = build_classifier().fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    overall = _scores(y_test, pred, prob)
    logger.info("Test accuracy %.4f | spam recall %.4f | ROC-AUC %.4f", overall["accuracy"], overall["recall"], overall["roc_auc"])

    fpr, tpr, _ = roc_curve(y_test, prob)
    prec, rec, _ = precision_recall_curve(y_test, prob)
    fpr, tpr = _downsample([fpr, tpr])
    rec, prec = _downsample([rec[::-1], prec[::-1]])

    sweep = []
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        p = (prob >= t).astype(int)
        sweep.append({"threshold": float(t), "precision": precision_score(y_test, p, zero_division=0),
                      "recall": recall_score(y_test, p, zero_division=0), "f1": f1_score(y_test, p, zero_division=0)})

    coef = feature_weights(model)
    names = vectorizer.get_feature_names_out()
    order = np.argsort(coef)
    top_spam = [{"term": pretty_term(names[i]), "weight": float(coef[i])} for i in order[::-1][:30]]
    top_ham = [{"term": pretty_term(names[i]), "weight": float(coef[i])} for i in order[:30]]

    errors = test_df.assign(prob=prob, pred=pred)
    errors = errors[errors["pred"] != errors["label"]]
    error_rows = [
        {"text": r.text[:400], "source": r.source, "actual": "Spam" if r.label else "Ham",
         "spam_probability": round(float(r.prob), 4)}
        for r in errors.sort_values("prob", key=lambda s: (s - 0.5).abs(), ascending=False).head(25).itertuples()
    ]

    metrics = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "vocabulary_size": len(names),
        "stop_words": len(STOP_WORDS),
        "data": {
            "total": len(df), "train": len(train_df), "test": len(test_df),
            "by_source": {s: {"ham": int(((df["source"] == s) & (df["label"] == 0)).sum()),
                              "spam": int(((df["source"] == s) & (df["label"] == 1)).sum())} for s in SOURCES},
        },
        "test": overall,
        "test_by_source": _by_source(test_df, pred, prob),
        "roc_curve": {"fpr": fpr, "tpr": tpr},
        "pr_curve": {"recall": rec, "precision": prec},
        "threshold_sweep": sweep,
        "test_probabilities": {"prob": np.round(prob, 4).tolist(), "label": y_test.tolist()},
        "top_spam_terms": top_spam,
        "top_ham_terms": top_ham,
        "misclassified": error_rows,
    }
    if benchmark:
        logger.info("Benchmarking the original pipeline and alternative models …")
        metrics["baseline"] = _baseline(train_df, test_df)
        metrics["model_comparison"] = _compare_models(X_train, y_train, X_test, y_test)
    metrics["training_seconds"] = round(time.perf_counter() - t_start, 1)

    if save:
        save_artifacts(model, vectorizer, metrics)
    return model, vectorizer, metrics


def save_artifacts(model, vectorizer, metrics):
    try:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        with open(VECTORIZER_PATH, "wb") as f:
            pickle.dump(vectorizer, f)
        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=1, default=float)
        logger.info("Saved model.pkl, vectorizer.pkl and metrics.json to %s", BASE_DIR)
    except OSError:
        logger.warning("Could not write artifacts (read-only filesystem?); using the in-memory model.")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    _, _, m = train_pipeline()
    print(f"\nTraining complete in {m['training_seconds']}s")
    print(f"{'Scope':8s} {'Acc':>7s} {'Prec':>7s} {'Recall':>7s} {'F1':>7s}")
    for scope, s in [("Overall", m["test"]), *m["test_by_source"].items()]:
        print(f"{scope:8s} {s['accuracy']:7.4f} {s['precision']:7.4f} {s['recall']:7.4f} {s['f1']:7.4f}")
    b = m["baseline"]["by_source"]["Email"]
    print(f"\nOriginal pipeline on email: recall {b['recall']:.4f}, F1 {b['f1']:.4f}")


if __name__ == "__main__":
    main()
