"""Rule-based risk signals shown alongside the ML verdict (independent of the model)."""

import re
from dataclasses import dataclass, field

from data_preprocessing import EMAIL as EMAIL_ADDR, MONEY, PERCENT, URL, extract_email_text

URGENCY = [
    "urgent", "immediately", "act now", "right away", "expires", "expire", "last chance",
    "limited time", "today only", "final notice", "within 24 hours", "asap", "suspended",
    "suspension", "avoid cancellation", "deadline", "don't miss", "hurry",
]
CREDENTIAL = [
    "verify your", "confirm your", "update your payment", "payment information", "payment details",
    "billing information", "bank details", "account details", "password", "login", "log in",
    "sign in", "social security", "ssn", "credit card", "card number", "pin number", "unlock your account",
]
PRIZE = [
    "winner", "won", "win ", "prize", "reward", "free", "gift card", "bonus", "cash", "lottery",
    "jackpot", "selected", "congratulations", "claim", "giveaway",
]
CALL_TO_ACTION = [
    "click here", "click below", "click the link", "tap here", "call now", "text ", "reply ",
    "sign up", "subscribe", "order now", "buy now", "apply now", "visit", "download",
]
BRANDS = [
    "amazon", "paypal", "apple", "icloud", "netflix", "microsoft", "google", "facebook", "instagram",
    "whatsapp", "bank", "irs", "fedex", "dhl", "ups", "usps", "ebay", "walmart", "hdfc", "sbi", "icici",
]
PHARMA = [
    "viagra", "cialis", "levitra", "pharmacy", "prescription", "pills", "weight loss", "lose weight",
    "diet pill", "enlargement", "replica", "casino", "dating", "singles", "xxx", "adult",
]
MONEY_WORDS = ["earn", "income", "investment", "loan", "mortgage", "credit", "debt", "commission", "refund"]
PHONE = re.compile(r"\b\d{5,}\b")
CAPS_WORD = re.compile(r"\b[A-Z]{3,}\b")
LETTER = re.compile(r"[A-Za-z]")


@dataclass
class Signal:
    key: str
    label: str
    value: float
    weight: float
    description: str
    matches: list = field(default_factory=list)

    @property
    def triggered(self) -> bool:
        return self.weight > 0


def _find_phrases(lower: str, phrases):
    return sorted({p.strip() for p in phrases if p in lower})


def extract_signals(raw: str) -> list:
    text = extract_email_text(raw)
    lower = f" {text.lower()} "
    letters = LETTER.findall(text)
    upper_ratio = sum(c.isupper() for c in letters) / len(letters) if letters else 0.0

    urls = list(dict.fromkeys(u.rstrip(">.,;:)\"'") for u in URL.findall(text)))
    emails = EMAIL_ADDR.findall(text)
    money = MONEY.findall(text)
    percents = PERCENT.findall(text)
    phones = PHONE.findall(text)
    caps = [w for w in CAPS_WORD.findall(text) if w not in {"USA", "FAQ", "PDF", "CEO", "HTML"}]
    exclaims = text.count("!")
    urgency = _find_phrases(lower, URGENCY)
    credential = _find_phrases(lower, CREDENTIAL)
    prize = _find_phrases(lower, PRIZE)
    cta = _find_phrases(lower, CALL_TO_ACTION)
    brands = [b for b in BRANDS if re.search(rf"\b{b}\b", lower)]
    money_words = _find_phrases(lower, MONEY_WORDS)
    pharma = _find_phrases(lower, PHARMA)

    def w(cond, val):
        return val if cond else 0

    return [
        Signal("links", "Links", len(urls), w(urls, min(20, 10 + 4 * len(urls))),
               "Embedded links are the main delivery mechanism for phishing and scams.", urls[:5]),
        Signal("credential", "Credential / payment request", len(credential), w(credential, 25),
               "Asks you to verify, log in or share payment details.", credential),
        Signal("urgency", "Urgency / pressure", len(urgency), w(urgency, min(20, 10 + 5 * len(urgency))),
               "Creates time pressure so you act before thinking.", urgency),
        Signal("prize", "Prize / free offer", len(prize), w(prize, min(20, 6 * len(prize))),
               "Promises rewards, prizes or free items.", prize),
        Signal("pharma", "Pharma / adult / gambling offer", len(pharma), w(pharma, min(25, 15 + 5 * len(pharma))),
               "Medication, adult, dating or casino offers are classic bulk-spam topics.", pharma),
        Signal("money", "Money amounts", len(money) + len(money_words), w(money or money_words, min(15, 5 * (len(money) + len(money_words)))),
               "Mentions cash amounts, earnings, loans or refunds.", (money + money_words)[:6]),
        Signal("cta", "Call to action", len(cta), w(cta, min(12, 4 * len(cta))),
               "Pushes you to click, call, reply or sign up.", cta),
        Signal("brand", "Brand mention", len(brands), w(brands and (credential or urls), 10),
               "Names a well-known company together with a link or credential request (impersonation).", brands),
        Signal("phone", "Phone / short codes", len(phones), w(phones, min(10, 5 * len(phones))),
               "Premium-rate numbers and SMS short codes are common in text spam.", phones[:5]),
        Signal("caps", "Shouting (ALL CAPS)", round(upper_ratio * 100, 1), w(upper_ratio > 0.3 or len(caps) >= 3, 8),
               "Heavy use of capital letters to grab attention.", caps[:6]),
        Signal("exclaim", "Exclamation marks", exclaims, w(exclaims >= 2, min(8, 2 * exclaims)),
               "Multiple exclamation marks signal hype.", []),
        Signal("percent", "Discounts / percentages", len(percents), w(percents, 5),
               "Percent-off claims are typical of promotional spam.", percents[:5]),
        Signal("emails", "Email addresses", len(emails), w(emails, 3),
               "Asks you to reply to an external address.", emails[:5]),
    ]


def risk_score(signals) -> int:
    return int(min(100, sum(s.weight for s in signals)))


def needs_review(spam_probability: float, signals) -> bool:
    """Rule layer: flag messages the model scores low but that show a phishing or strong spam pattern."""
    return is_phishing_like(signals) or (spam_probability >= 0.25 and risk_score(signals) >= 25)


def is_phishing_like(signals) -> bool:
    by = {s.key: s for s in signals}
    return by["credential"].triggered and (
        by["urgency"].triggered or by["links"].triggered or by["brand"].triggered or by["cta"].triggered
    )
