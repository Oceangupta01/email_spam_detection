"""High-level detector used by the Streamlit UI: ML probability + rule-based risk signals + explanations."""

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import predict
from data_preprocessing import extract_email_text, pretty_term, transform_text

from .signals import extract_signals, needs_review, risk_score

DEFAULT_THRESHOLD = predict.DEFAULT_THRESHOLD
_CHUNK = re.compile(r"\S+|\s+")


@dataclass
class Analysis:
    text: str
    spam_probability: float
    threshold: float
    verdict: str
    risk_score: int
    phishing_like: bool
    signals: list
    contributions: list
    highlights: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def risk_level(self) -> str:
        score = max(self.spam_probability * 100, self.risk_score if self.phishing_like else 0)
        if score >= 75:
            return "High"
        if score >= 40:
            return "Medium"
        return "Low"


def _verdict(prob: float, threshold: float, flagged: bool) -> str:
    if prob >= threshold:
        return "Spam"
    return "Suspicious" if flagged else "Not Spam"


class SpamDetector:
    def __init__(self):
        self.model, self.vectorizer, self.metrics, self.coef = predict.load_artifacts()
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.vocab = self.vectorizer.vocabulary_

    def predict_proba(self, texts) -> np.ndarray:
        return predict.spam_probabilities(texts)

    def contributions(self, text: str, top_n: int = 12) -> list:
        row = self.vectorizer.transform([transform_text(extract_email_text(text))])
        contrib = row.data * self.coef[row.indices]
        order = np.argsort(-np.abs(contrib))[:top_n]
        return [{"term": pretty_term(self.feature_names[row.indices[i]]), "contribution": float(contrib[i])} for i in order]

    def highlight(self, text: str) -> list:
        """Split text into (chunk, weight) pairs; weight is the summed unigram weight of the chunk."""
        out = []
        for chunk in _CHUNK.findall(text):
            if chunk.isspace():
                out.append((chunk, 0.0))
                continue
            toks = transform_text(chunk).split()
            out.append((chunk, float(sum(self.coef[self.vocab[t]] for t in toks if t in self.vocab))))
        return out

    def analyze(self, text: str, threshold: float = DEFAULT_THRESHOLD) -> Analysis:
        text = text[: predict.MAX_INPUT_LENGTH]
        prob = float(self.predict_proba([text])[0])
        signals = extract_signals(text)
        phishing = needs_review(prob, signals)
        words = re.findall(r"\b\w+\b", text)
        stats = {
            "characters": len(text),
            "words": len(words),
            "sentences": max(1, len(re.findall(r"[.!?]+(?:\s|$)", text))),
            "avg_word_length": round(float(np.mean([len(w) for w in words])), 1) if words else 0,
            "known_terms": int(self.vectorizer.transform([transform_text(extract_email_text(text))]).nnz),
        }
        return Analysis(
            text=text, spam_probability=prob, threshold=threshold, verdict=_verdict(prob, threshold, phishing),
            risk_score=risk_score(signals), phishing_like=phishing, signals=signals,
            contributions=self.contributions(text), highlights=self.highlight(text), stats=stats,
        )

    def analyze_batch(self, texts, threshold: float = DEFAULT_THRESHOLD, progress=None) -> pd.DataFrame:
        texts = [str(t) for t in texts]
        probs = self.predict_proba(texts)
        rows = []
        for i, (text, prob) in enumerate(zip(texts, probs)):
            signals = extract_signals(text[: predict.MAX_INPUT_LENGTH])
            phishing = needs_review(prob, signals)
            rows.append({
                "message": text, "verdict": _verdict(prob, threshold, phishing), "spam_probability": float(prob),
                "risk_score": risk_score(signals), "phishing_pattern": phishing, "words": len(text.split()),
                "signals": ", ".join(s.label for s in sorted(signals, key=lambda s: -s.weight) if s.triggered),
            })
            if progress and (i % 50 == 0 or i == len(texts) - 1):
                progress((i + 1) / len(texts))
        return pd.DataFrame(rows)
