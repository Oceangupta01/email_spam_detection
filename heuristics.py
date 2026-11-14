"""
Rule-based heuristic threat scoring engine.

Provides an independent risk signal (0.0 – 1.0) derived from
structural and lexical patterns in the raw message text.

This is used as a second layer alongside the ML classifier to catch
modern email spam patterns that fall outside the original SMS training
distribution (phishing, 419 advance-fee fraud, pharmaceutical spam,
crypto scams, work-from-home scams, etc.).
"""

import re
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Pattern Groups (each returns a score contribution and a threat label)
# ---------------------------------------------------------------------------

_PATTERNS: List[Tuple[str, float, str]] = [
    # ---- Urgency / Pressure ----
    (r'\b(urgent|urgently|act now|immediately|expire[sd]?|last chance|final notice|within \d+ hours?|limited time)\b', 0.25, "Urgency Pressure"),
    (r'!{2,}', 0.15, "Excessive Exclamation"),

    # ---- Financial Solicitation ----
    (r'[\$£€¥₹]\s*\d[\d,\.]*', 0.20, "Currency Mention"),
    (r'\b(\d[\d,\.]+\s*(?:USD|GBP|EUR|dollars?|pounds?|euros?))\b', 0.20, "Financial Solicitation"),
    (r'\b(wire transfer|bank transfer|western union|money gram|bitcoin transfer|send money|transfer funds?)\b', 0.40, "Payment Method"),

    # ---- 419 / Advance-Fee Fraud ----
    (r'\b(nigerian?|barrister|deceased client|inheritance|next of kin|diplomat|consignment|confidential deal)\b', 0.45, "419 Advance-Fee Fraud"),
    (r'\b(million(?:s)? (?:dollars?|usd|pounds?|euros?))\b', 0.35, "Large Amount Promise"),
    (r'\b(transfer.{0,30}(?:account|fund)|fund.{0,30}transfer|offshore account|foreign account)\b', 0.45, "419 Advance-Fee Fraud"),
    (r'\b(prince|princess|king|queen).{0,50}(?:million|transfer|fund|account|money)\b', 0.55, "419 Royalty Scam"),

    # ---- Phishing / Account Takeover ----
    (r'\b(account (?:suspended|locked|compromised|restricted|terminated|disabled))\b', 0.40, "Account Threat"),
    (r'\b(verify (?:your )?(?:account|identity|email|phone|password|details?))\b', 0.35, "Credential Phishing"),
    (r'\b(click (?:here|below|the link)|click to (?:verify|confirm|unlock|restore|activate))\b', 0.30, "Suspicious CTA"),
    (r'\b(paypal|apple\s*id|amazon|netflix|microsoft|google)\b.*\b(locked|suspended|verify|compromised|unauthorized)\b', 0.45, "Brand Phishing"),
    (r'\b(bank account.{0,40}(?:closed|suspended|update|verify)|update.{0,20}(?:your )?(?:bank|account|details?))\b', 0.45, "Bank Account Threat"),

    # ---- Crypto / Investment Scams ----
    (r'\b(bitcoin|btc|ethereum|eth|crypto|nft|blockchain|private key|seed phrase|wallet)\b', 0.20, "Crypto Mention"),
    (r'\b(double your|guaranteed returns?|risk.?free investment|passive income)\b', 0.40, "Investment Scam"),
    (r'\b(wallet.{0,30}(?:compromised|hacked|flagged|locked)|seed phrase|private key)\b', 0.60, "Crypto Wallet Threat"),
    (r'\b(send.{0,10}(?:btc|eth|bitcoin|crypto).{0,10}(?:receive|get|back|return)|double.{0,10}(?:btc|eth|bitcoin))\b', 0.65, "Crypto Doubling Scam"),

    # ---- Pharmaceutical Spam ----
    (r'\b(viagra|cialis|levitra|sildenafil|generic pills?|online pharmacy|no prescription|cheap meds?)\b', 0.50, "Pharmaceutical Spam"),
    (r'\b(buy.{0,20}(?:pills?|meds?|drugs?|medications?)|pills?.{0,20}(?:online|cheap|discount|no prescription))\b', 0.50, "Pharmaceutical Spam"),

    # ---- Employment Scam ----
    (r'\b(work from home|earn (?:\$|£|€)\d+|make (?:\$|£|€)\d+|daily income|no experience (?:required|needed))\b', 0.35, "Employment Scam"),
    (r'\b(earn \d+ (?:dollars?|per day|daily|weekly))\b', 0.35, "Income Claim"),

    # ---- Lottery / Prize Scams ----
    (r'\b(you(?:\'ve)? won|winner|jackpot|lottery|prize claim|gift card|selected to receive)\b', 0.40, "Prize / Lottery Scam"),
    (r'\b(claim your (?:prize|reward|gift|winnings?))\b', 0.40, "Claim Trigger"),

    # ---- Suspicious Links / Infrastructure ----
    (r'https?://(?:bit\.ly|t\.co|tinyurl|is\.gd|goo\.gl|ow\.ly|rb\.gy|short|tiny|clck)\b', 0.35, "URL Shortener"),
    (r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', 0.45, "IP-Based URL"),
    (r'(?:paypal|amazon|apple|microsoft|google|netflix)[^.]*\.(?!com|co\.uk|org|net)\w+', 0.45, "Homograph Domain"),

    # ---- Obfuscation / Structural ----
    (r'[A-Z]{4,}(?:\s+[A-Z]{4,}){2,}', 0.20, "Excessive Capitalization"),
    (r'\b(free(?:msg)?|f\.?r\.?e\.?e)\b', 0.20, "Free Keyword"),
    (r'\b(guaranteed|100%|absolutely free|no cost|risk.?free)\b', 0.20, "Guarantee Language"),
]

# Pre-compile all patterns for performance
_COMPILED = [(re.compile(pat, re.IGNORECASE), score, label) for pat, score, label in _PATTERNS]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_heuristic_score(text: str) -> Tuple[float, List[str]]:
    """Compute a rule-based spam risk score for the raw input text.

    Parameters
    ----------
    text : str
        Original (un-preprocessed) message text.

    Returns
    -------
    Tuple[float, List[str]]
        - ``score``  : float in [0.0, 1.0] — higher = more likely spam
        - ``signals``: list of triggered threat-label strings for explainability
    """
    score = 0.0
    triggered: List[str] = []

    for pattern, weight, label in _COMPILED:
        if pattern.search(text):
            score += weight
            if label not in triggered:
                triggered.append(label)

    # Clamp to [0, 1]
    return min(score, 1.0), triggered


def fuse_scores(ml_spam_prob: float, heuristic_score: float, ml_weight: float = 0.55) -> float:
    """Combine ML probability and heuristic score into a final spam probability.

    Strategy:
    - If heuristic_score >= 0.65 (strong rule-based signals) → hard floor of 0.55 (spam)
    - If ml_spam_prob >= 0.70 (confident ML spam) → hard floor of 0.55 (spam)
    - Otherwise: weighted probabilistic OR fusion

    Parameters
    ----------
    ml_spam_prob : float
        Spam probability from the ML classifier (0.0 – 1.0).
    heuristic_score : float
        Rule-based risk score from ``compute_heuristic_score`` (0.0 – 1.0).
    ml_weight : float
        Weight applied to the ML probability term (default 0.55).

    Returns
    -------
    float
        Combined spam probability in [0.0, 1.0].
    """
    h_weight = 1.0 - ml_weight
    fused = 1.0 - (1.0 - ml_weight * ml_spam_prob) * (1.0 - h_weight * heuristic_score)

    # Hard override: if either signal is very confident alone, declare spam
    if heuristic_score >= 0.65:
        fused = max(fused, 0.60)
    if ml_spam_prob >= 0.70:
        fused = max(fused, 0.60)

    return min(fused, 1.0)
