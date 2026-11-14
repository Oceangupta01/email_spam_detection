"""
Shared text preprocessing for the Email/SMS Spam Detection system.

This module is used by both training and prediction to ensure
consistent text transformation.

v2: Enhanced with synthetic placeholder tokens to preserve high-signal
spam indicators (URLs, money amounts, email addresses, excessive caps)
that were previously stripped out by the alphanumeric filter, causing
many real-world email spam messages to be misclassified as ham.
"""

import re
import string
import logging

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NLTK bootstrap – download required data once
# ---------------------------------------------------------------------------

def ensure_nltk_data() -> None:
    """Download NLTK resources if they are not already present."""
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    for resource in ("punkt", "punkt_tab", "stopwords"):
        try:
            nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
        except LookupError:
            logger.info("Downloading NLTK resource: %s", resource)
            nltk.download(resource, quiet=True)

ensure_nltk_data()

# Instantiate stemmer once at module level
_stemmer = PorterStemmer()
_stop_words = set(stopwords.words("english"))
_punctuation = set(string.punctuation)

# ---------------------------------------------------------------------------
# Regex patterns for placeholder injection (run BEFORE lowercasing)
# ---------------------------------------------------------------------------
_RE_URL = re.compile(
    r'https?://\S+|www\.\S+|(?:[a-zA-Z0-9-]+\.)+(?:com|net|org|info|biz|io|co|xyz|club|site|click|tk|ml|ga|cf|win|loan|top|pw|ru|cn)\b',
    re.IGNORECASE,
)
_RE_EMAIL_ADDR = re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b')
_RE_MONEY = re.compile(r'(?:[\$£€¥₹]\s*\d[\d,\.]*|\d[\d,\.]*\s*(?:USD|GBP|EUR|dollars?|pounds?|euros?|BTC|ETH|USDT))', re.IGNORECASE)
_RE_PHONE = re.compile(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b|\b09\d{8,10}\b|\b08\d{8,10}\b')
_RE_EXCLAIM = re.compile(r'!{2,}')   # multiple exclamation marks
_RE_CAPS_WORD = re.compile(r'\b[A-Z]{3,}\b')  # all-caps words ≥3 letters


def _inject_placeholders(text: str) -> str:
    """Replace high-signal spans with synthetic placeholder tokens before cleaning.

    This preserves critical spam signals that the alphanumeric tokeniser
    would otherwise silently discard.
    """
    # Count uppercase words before any substitution
    caps_words = _RE_CAPS_WORD.findall(text)
    caps_count = len(caps_words)

    text = _RE_URL.sub(' __url__ ', text)
    text = _RE_EMAIL_ADDR.sub(' __emailaddr__ ', text)
    text = _RE_MONEY.sub(' __money__ ', text)
    text = _RE_PHONE.sub(' __phone__ ', text)
    text = _RE_EXCLAIM.sub(' __exclaim__ ', text)

    if caps_count >= 2:
        # Append urgency tokens proportional to caps usage (capped at 3)
        text += (' __urgentcaps__' * min(caps_count, 3))

    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transform_text(text: str) -> str:
    """Clean and normalise a single text string.

    Steps
    -----
    1. Inject semantic placeholder tokens (URLs, money, emails, caps)
    2. Lower-case
    3. Tokenise (NLTK word_tokenize)
    4. Keep alphanumeric tokens (placeholders already alphanumeric)
    5. Remove English stop-words
    6. Apply Porter stemming

    Parameters
    ----------
    text : str
        Raw email / SMS body text.

    Returns
    -------
    str
        Space-joined cleaned tokens including placeholder signals.
    """
    if not isinstance(text, str):
        text = str(text)

    # Step 1: inject before case-folding so caps detection works
    text = _inject_placeholders(text)

    # Step 2: lowercase
    text = text.lower()

    # Step 3: tokenise
    tokens = nltk.word_tokenize(text)

    # Step 4: keep alphanumeric tokens (placeholders like __url__ are alnum-with-underscore)
    tokens = [t for t in tokens if t.isalnum() or t.startswith('__')]

    # Step 5: remove stop-words (but never remove our placeholders)
    tokens = [
        t for t in tokens
        if t.startswith('__') or (t not in _stop_words and t not in _punctuation)
    ]

    # Step 6: stem (placeholders are skipped by the stemmer gracefully)
    tokens = [t if t.startswith('__') else _stemmer.stem(t) for t in tokens]

    return " ".join(tokens)
