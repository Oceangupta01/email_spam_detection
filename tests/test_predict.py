"""Tests for the predict module."""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from predict import predict


class TestPredict:
    """Integration tests for the predict() function."""

    def test_predict_returns_dict(self):
        result = predict("Hello, how are you?")
        assert isinstance(result, dict)
        assert "label" in result
        assert "confidence" in result

    def test_label_values(self):
        result = predict("Free prize money click now")
        assert result["label"] in ("spam", "ham")

    def test_confidence_range(self):
        result = predict("Win a million dollars immediately")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_spam_detection(self):
        """A clearly spammy message should be classified as spam."""
        result = predict(
            "WINNER!! You have been selected to receive a $1000 cash prize! "
            "Call 09061701461 to claim NOW!"
        )
        assert result["label"] == "spam"

    def test_ham_detection(self):
        """A normal message should be classified as ham."""
        result = predict("Hey, are we still meeting for lunch tomorrow?")
        assert result["label"] == "ham"

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="empty"):
            predict("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="empty"):
            predict("   ")

    def test_very_long_input_raises(self):
        long_text = "a" * 20_000
        with pytest.raises(ValueError, match="maximum length"):
            predict(long_text)


EMAIL_SPAM = [
    "Congratulations! You have won a $1,000 Walmart gift card. Click here to claim your reward "
    "before it expires: http://bit.ly/claim-now",
    "I am a barrister representing the estate of a late client who left $15.5 million USD. I need your "
    "assistance to transfer the funds; you will receive 30% for your help. Reply with your bank details.",
    "Get approved for a personal loan up to $50,000 today regardless of credit score. Limited time offer, apply now!",
    "Earn $5000 per week from home with our bitcoin trading bot. Guaranteed returns. Sign up now, limited spots available!",
]

EMAIL_HAM = [
    "Hi team, attached is the agenda for Monday's meeting. Please review the quarterly report before then.",
    "Your Amazon order #112-3345 has shipped and will arrive on Thursday.",
]


@pytest.mark.parametrize("text", EMAIL_SPAM)
def test_email_spam_detected(text):
    assert predict(text)["label"] == "spam"


@pytest.mark.parametrize("text", EMAIL_HAM)
def test_email_ham_not_flagged(text):
    assert predict(text)["label"] == "ham"


def test_threshold_changes_label():
    text = "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry"
    prob = predict(text)["spam_probability"]
    assert predict(text, threshold=min(prob + 0.01, 1.0))["label"] == "ham"
    assert predict(text, threshold=max(prob - 0.01, 0.0))["label"] == "spam"


def test_detector_flags_pharma_as_suspicious():
    from spam_model import SpamDetector

    analysis = SpamDetector().analyze(
        "Buy cheap viagra and cialis online, no prescription needed. 80% discount, discreet shipping worldwide."
    )
    assert analysis.verdict in ("Spam", "Suspicious")
