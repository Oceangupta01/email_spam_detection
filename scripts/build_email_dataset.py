"""
Build data/emails.csv.gz from the public SpamAssassin corpus.

Usage:
    python scripts/build_email_dataset.py
"""

import email
import html
import io
import os
import re
import tarfile
import urllib.request
from email import policy

import pandas as pd

BASE_URL = "https://spamassassin.apache.org/old/publiccorpus/"
ARCHIVES = {
    "20030228_easy_ham.tar.bz2": 0,
    "20030228_hard_ham.tar.bz2": 0,
    "20030228_spam.tar.bz2": 1,
    "20050311_spam_2.tar.bz2": 1,
}
OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "emails.csv.gz")
MAX_CHARS = 10_000


def _message_text(raw: bytes) -> str:
    msg = email.message_from_bytes(raw, policy=policy.default)
    parts = []
    for part in msg.walk():
        if part.get_content_maintype() != "text":
            continue
        try:
            body = part.get_content()
        except Exception:
            body = (part.get_payload(decode=True) or b"").decode("latin-1", "ignore")
        if part.get_content_subtype() == "html":
            body = re.sub(r"<(script|style).*?</\1>", " ", body, flags=re.S | re.I)
            body = html.unescape(re.sub(r"<[^>]+>", " ", body))
        parts.append(body)
    text = f"{msg.get('Subject', '')} {' '.join(parts)}"
    return re.sub(r"\s+", " ", text).strip()[:MAX_CHARS]


def main():
    rows = []
    for name, label in ARCHIVES.items():
        print(f"Downloading {name} …")
        with urllib.request.urlopen(BASE_URL + name) as resp:
            data = resp.read()
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:bz2") as tar:
            for member in tar.getmembers():
                if not member.isfile() or member.name.endswith("cmds"):
                    continue
                text = _message_text(tar.extractfile(member).read())
                if text:
                    rows.append((label, text))

    df = pd.DataFrame(rows, columns=["target", "text"]).drop_duplicates("text")
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False, compression="gzip")
    print(f"Saved {len(df)} emails ({df.target.sum()} spam) to {OUT_PATH}")


if __name__ == "__main__":
    main()
