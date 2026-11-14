"""
Shared text preprocessing for the Email/SMS Spam Detection system.

Used by both training and prediction so the transformation is identical.
Spam signals (links, money, percentages, phone numbers, "!", ALL-CAPS)
are converted to placeholder tokens instead of being thrown away.
"""

import email
import html
import re
from email import policy

from nltk.stem.porter import PorterStemmer

# NLTK's English stop-word list, embedded so no corpus download is needed at runtime.
STOP_WORDS = frozenset("""
a about above after again against ain all am an and any are aren as at be because been
before being below between both but by can couldn d did didn do does doesn doing don down
during each few for from further had hadn has hasn have haven having he her here hers
herself him himself his how i if in into is isn it its itself just ll m ma me mightn more
most mustn my myself needn no nor not now o of off on once only or other our ours ourselves
out over own re s same shan she should shouldn so some such t than that the their theirs
them themselves then there these they this those through to too under until up ve very was
wasn we were weren what when where which while who whom why will with won wouldn y you your
yours yourself yourselves
""".split())

SIGNAL_TOKENS = {
    "emailtoken": "email address",
    "urltoken": "link",
    "moneytoken": "money amount",
    "percenttoken": "percentage",
    "phonetoken": "phone number",
    "exclamtoken": "!",
    "capstoken": "ALL-CAPS word",
}

EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
URL = re.compile(
    r"(https?://\S+|www\.\S+|\b[\w-]+\.(?:com|net|org|info|biz|ru|ly|co|io|xyz|top)\b\S*)",
    re.I,
)
MONEY = re.compile(
    r"[$£€]\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s?(?:usd|dollars|pounds|gbp|eur|euros)\b",
    re.I,
)
PERCENT = re.compile(r"\b\d+(?:\.\d+)?\s?%")
PHONE = re.compile(r"\+?\d[\d\s-]{8,}\d")
CAPS = re.compile(r"\b[A-Z]{3,}\b")
_WORD = re.compile(r"[a-z0-9]+")

_stemmer = PorterStemmer()
_MAX_REPEAT = 3


def transform_text(text) -> str:
    """Normalise a raw email/SMS into space-joined stemmed tokens plus signal tokens."""
    if not isinstance(text, str):
        text = str(text)

    caps = min(len(CAPS.findall(text)), _MAX_REPEAT)
    exclam = min(text.count("!"), _MAX_REPEAT)

    text = EMAIL.sub(" emailtoken ", text)
    text = URL.sub(" urltoken ", text)
    text = MONEY.sub(" moneytoken ", text)
    text = PERCENT.sub(" percenttoken ", text)
    text = PHONE.sub(" phonetoken ", text)

    tokens = [
        t if t in SIGNAL_TOKENS else _stemmer.stem(t)
        for t in _WORD.findall(text.lower())
        if t not in STOP_WORDS
    ]
    tokens += ["exclamtoken"] * exclam + ["capstoken"] * caps
    return " ".join(tokens)


_HEADER_LINE = re.compile(r"^[A-Za-z][A-Za-z0-9-]{1,40}:\s?.*$")
_HTML_TAG = re.compile(r"<[a-zA-Z!/][^>]*>")
_SCRIPT_STYLE = re.compile(r"<(script|style)\b.*?</\1>", re.S | re.I)
_HREF = re.compile(r"""href\s*=\s*["']?([^"'\s>]+)""", re.I)


def _looks_like_raw_email(text: str) -> bool:
    lines = [ln for ln in text.lstrip().splitlines()[:15] if ln.strip()]
    return bool(lines) and _HEADER_LINE.match(lines[0]) is not None and sum(
        bool(_HEADER_LINE.match(ln)) for ln in lines
    ) >= 2


def strip_html(text: str) -> str:
    links = " ".join(_HREF.findall(text))
    text = _HTML_TAG.sub(" ", _SCRIPT_STYLE.sub(" ", text))
    return f"{html.unescape(text)} {links}"


def extract_email_text(text: str) -> str:
    """Return sender + subject + body for raw RFC-822 emails / HTML; other text is returned unchanged."""
    if _looks_like_raw_email(text):
        msg = email.message_from_string(text, policy=policy.default)
        part = msg.get_body(preferencelist=("plain", "html"))
        if part is not None:
            try:
                body = part.get_content()
            except (LookupError, ValueError):
                body = str(part.get_payload())
            text = f"{msg.get('From', '') or ''}\n{msg.get('Subject', '') or ''}\n{body}"
    if _HTML_TAG.search(text):
        text = strip_html(text)
    return text


def pretty_term(term: str) -> str:
    """Human-readable form of a vocabulary term, e.g. 'urltoken now' -> '[link] now'."""
    return " ".join(f"[{SIGNAL_TOKENS[w]}]" if w in SIGNAL_TOKENS else w for w in term.split())
