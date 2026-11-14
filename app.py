"""
Streamlit Enterprise Dashboard for the Email/SMS Spam Detection system.

Usage:
    streamlit run app.py

v2: Professional multi-tab dashboard with:
  - Dynamic Plotly charts (gauge, radar, waterfall)
  - Text forensics panel
  - Quick preset loader
  - Batch CSV/file scanner
  - Model comparison & explainability tab
  - Dataset analytics tab
"""

import io
import re
import textwrap
from collections import Counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Page config must be first ──────────────────────────────────────────────
st.set_page_config(
    page_title="Sentinel Spam Shield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Lazy imports ───────────────────────────────────────────────────────────
from predict import predict  # noqa: E402

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Global */
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"] { background: #161b27; border-right: 1px solid #2a3045; }
h1,h2,h3,h4 { color: #e2e8f0 !important; }
p, li, label { color: #94a3b8 !important; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1e2535, #252d42);
    border: 1px solid #2a3045;
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
    margin-bottom: 10px;
}
.metric-card .metric-value { font-size: 2rem; font-weight: 800; }
.metric-card .metric-label { font-size: 0.8rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; }
.spam-card { border-color: #ef4444 !important; background: linear-gradient(135deg, #2a0e0e, #3f1515) !important; }
.ham-card  { border-color: #22c55e !important; background: linear-gradient(135deg, #0d2a16, #133820) !important; }

/* Signal pill badges */
.signal-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    margin: 3px 3px;
    border: 1px solid #3a4a60;
    background: #1a2235;
    color: #94a3b8;
}
.signal-high { background: #3f1515; border-color: #ef4444; color: #fca5a5; }
.signal-med  { background: #3f2e0a; border-color: #f59e0b; color: #fcd34d; }

/* Highlight spans */
.hl-spam { background: rgba(239,68,68,0.25); border-radius: 4px; padding: 1px 3px; color: #fca5a5; font-weight: 600; }
.hl-sus  { background: rgba(245,158,11,0.20); border-radius: 4px; padding: 1px 3px; color: #fcd34d; }

/* Text area */
textarea { background: #1e2535 !important; color: #e2e8f0 !important; border: 1px solid #2a3045 !important; }

/* Tabs */
[data-testid="stTabs"] [role="tab"] { color: #64748b !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: #3b82f6 !important; border-bottom: 2px solid #3b82f6 !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
.stButton > button:hover { background: linear-gradient(135deg, #3b82f6, #2563eb) !important; }

/* Divider */
hr { border-color: #2a3045 !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants & Presets ────────────────────────────────────────────────────
PRESETS = {
    "🎣 PayPal Phishing": "URGENT: Your PayPal account has been temporarily suspended due to unusual activity detected from an unrecognized device. To restore full access, please verify your identity immediately by clicking the link below: http://paypal-account-verify-login.com/secure. Failure to verify within 24 hours will result in permanent account closure.",
    "👑 Nigerian Prince (419)": "Dear Friend, I am Barrister Johnson Adeyemi, a legal practitioner and I represent the estate of a deceased client who shared your surname. He left behind $15.5 million USD in an offshore account with no next of kin. I need your cooperation to transfer these funds. You will receive 30% as your share. Please reply with your full name and bank account details.",
    "💊 Pharma Spam": "BEST PRICES on Generic Medications! No Prescription Required! Buy Viagra 100mg, Cialis 20mg, Levitra at 70-80% off retail price. Fast discreet shipping worldwide. FDA approved generics. Visit our secure online pharmacy now and use code SAVE50 for an extra 50% discount on your first order!",
    "🪙 Crypto Scam": "🚀 BITCOIN GIVEAWAY IS LIVE! Elon Musk is sending back double the Bitcoin to celebrate Tesla's record profits! Send 0.1 BTC to 1A2b3C4d5E6f and receive 0.2 BTC back within 60 minutes! GUARANTEED! This offer expires in 30 minutes - only 47 slots remaining! Act NOW!!!",
    "💼 Work-From-Home Scam": "URGENT HIRING: Earn $500-$1,000 per day doing simple data entry tasks from home! No experience required. No degree needed. Work flexible hours. Daily PayPal payments. We are currently filling 50 positions immediately. Click here to apply: http://earnfromhome-jobs.com/apply?ref=immediate",
    "🏦 Invoice Scam (BEC)": "URGENT ACTION REQUIRED: Invoice #INV-847203 for $4,850.00 is now 60 days past due. Please process the wire transfer to our NEW banking details immediately (we have changed banks). Failure to pay within 24 hours will result in legal action and referral to our collections agency. Download the invoice attachment and confirm payment.",
    "📧 Legitimate Business": "Hi Team, please find attached the Q3 financial performance report for review before our Tuesday board meeting at 2pm. The key metrics look strong this quarter. Please send any questions or comments to me by end of day Monday. Thanks and see you all on Tuesday.",
    "💬 Legitimate SMS": "Hey! Are you free for dinner tomorrow night? Thinking of trying that new Italian place on 5th. Let me know if you're in and I'll make a reservation for 7pm 🍝",
}

THREAT_AXES = ["Urgency Pressure", "Financial Bait", "Credential Theft", "Suspicious Links", "Obfuscation"]

SIGNAL_TO_AXIS = {
    "Urgency Pressure": 0, "Excessive Exclamation": 0,
    "Currency Mention": 1, "Financial Solicitation": 1, "Payment Method": 1,
    "419 Advance-Fee Fraud": 1, "Large Amount Promise": 1,
    "Account Threat": 2, "Credential Phishing": 2, "Brand Phishing": 2,
    "Suspicious CTA": 2, "Bank Account Threat": 2,
    "URL Shortener": 3, "IP-Based URL": 3, "Homograph Domain": 3,
    "Excessive Capitalization": 4, "Free Keyword": 4, "Guarantee Language": 4,
    "Pharmaceutical Spam": 1, "Employment Scam": 1, "Income Claim": 1,
    "Prize / Lottery Scam": 1, "Claim Trigger": 1,
    "Crypto Mention": 3, "Investment Scam": 1,
    "Crypto Wallet Threat": 2, "Crypto Doubling Scam": 1,
    "419 Royalty Scam": 1,
}

# ── Chart helpers ──────────────────────────────────────────────────────────

def make_gauge(spam_prob: float) -> go.Figure:
    pct = spam_prob * 100
    if pct < 35:
        color, label = "#22c55e", "SAFE"
    elif pct < 60:
        color, label = "#f59e0b", "SUSPICIOUS"
    else:
        color, label = "#ef4444", "SPAM DETECTED"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 42, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#4a5568"},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "#1e2535",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 35],  "color": "#0d2a16"},
                {"range": [35, 60], "color": "#2d1f06"},
                {"range": [60, 100],"color": "#2a0e0e"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.8,
                "value": pct,
            },
        },
        title={"text": f"<b>{label}</b>", "font": {"size": 16, "color": color}},
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        font={"color": "#94a3b8"},
        margin=dict(t=60, b=20, l=20, r=20),
        height=280,
    )
    return fig


def make_radar(threat_signals: list, heuristic_score: float) -> go.Figure:
    # Map signals to axis scores
    axis_scores = [0.0] * 5
    signal_weights = {
        "Urgency Pressure": 0.8, "Excessive Exclamation": 0.5,
        "Currency Mention": 0.4, "Financial Solicitation": 0.6,
        "Payment Method": 0.9, "419 Advance-Fee Fraud": 0.9,
        "Large Amount Promise": 0.7, "Account Threat": 0.9,
        "Credential Phishing": 0.85, "Brand Phishing": 0.9,
        "Suspicious CTA": 0.7, "Bank Account Threat": 0.8,
        "URL Shortener": 0.7, "IP-Based URL": 0.9, "Homograph Domain": 0.95,
        "Excessive Capitalization": 0.5, "Free Keyword": 0.4,
        "Guarantee Language": 0.5, "Pharmaceutical Spam": 0.9,
        "Employment Scam": 0.8, "Income Claim": 0.7,
        "Prize / Lottery Scam": 0.85, "Claim Trigger": 0.75,
        "Crypto Mention": 0.5, "Investment Scam": 0.8,
        "Crypto Wallet Threat": 0.9, "Crypto Doubling Scam": 0.95,
        "419 Royalty Scam": 0.9,
    }
    for sig in threat_signals:
        ax = SIGNAL_TO_AXIS.get(sig, -1)
        if ax >= 0:
            axis_scores[ax] = max(axis_scores[ax], signal_weights.get(sig, 0.5))

    # Blend with overall heuristic
    if heuristic_score > 0 and all(s == 0 for s in axis_scores):
        axis_scores = [heuristic_score * 0.5] * 5

    vals = axis_scores + [axis_scores[0]]  # close polygon
    cats = THREAT_AXES + [THREAT_AXES[0]]

    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=cats,
        fill="toself",
        fillcolor="rgba(239, 68, 68, 0.15)",
        line=dict(color="#ef4444", width=2),
        name="Threat",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#1e2535",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#2a3045", tickcolor="#4a5568", tickfont=dict(size=9, color="#64748b")),
            angularaxis=dict(gridcolor="#2a3045", linecolor="#2a3045", tickfont=dict(size=11, color="#94a3b8")),
        ),
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        showlegend=False,
        margin=dict(t=30, b=30, l=50, r=50),
        height=300,
    )
    return fig


def make_confidence_bars(ml_prob: float, heuristic_score: float, final_prob: float) -> go.Figure:
    categories = ["ML Model", "Heuristic Engine", "Final (Combined)"]
    values = [ml_prob * 100, heuristic_score * 100, final_prob * 100]
    colors = ["#3b82f6", "#f59e0b", "#ef4444" if final_prob >= 0.5 else "#22c55e"]

    fig = go.Figure(go.Bar(
        x=values, y=categories, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=13),
    ))
    fig.update_layout(
        paper_bgcolor="#0f1117", plot_bgcolor="#1e2535",
        xaxis=dict(range=[0, 115], gridcolor="#2a3045", color="#64748b", ticksuffix="%"),
        yaxis=dict(color="#94a3b8"),
        margin=dict(t=10, b=10, l=10, r=70),
        height=170,
        font=dict(color="#94a3b8"),
    )
    return fig


def make_signal_hbar(threat_signals: list) -> go.Figure | None:
    if not threat_signals:
        return None
    signal_weights = {
        "Urgency Pressure": 0.8, "Excessive Exclamation": 0.5,
        "Currency Mention": 0.4, "Financial Solicitation": 0.6,
        "Payment Method": 0.9, "419 Advance-Fee Fraud": 0.9,
        "Large Amount Promise": 0.7, "Account Threat": 0.9,
        "Credential Phishing": 0.85, "Brand Phishing": 0.9,
        "Suspicious CTA": 0.7, "Bank Account Threat": 0.8,
        "URL Shortener": 0.7, "IP-Based URL": 0.9, "Homograph Domain": 0.95,
        "Excessive Capitalization": 0.5, "Free Keyword": 0.4,
        "Guarantee Language": 0.5, "Pharmaceutical Spam": 0.9,
        "Employment Scam": 0.8, "Income Claim": 0.7,
        "Prize / Lottery Scam": 0.85, "Claim Trigger": 0.75,
        "Crypto Mention": 0.5, "Investment Scam": 0.8,
        "Crypto Wallet Threat": 0.9, "Crypto Doubling Scam": 0.95,
        "419 Royalty Scam": 0.9,
    }
    vals = [signal_weights.get(s, 0.5) for s in threat_signals]
    colors = ["#ef4444" if v >= 0.7 else "#f59e0b" for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=threat_signals, orientation="h",
        marker=dict(color=colors),
        text=[f"{v*100:.0f}%" for v in vals],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=11),
    ))
    fig.update_layout(
        paper_bgcolor="#0f1117", plot_bgcolor="#1e2535",
        xaxis=dict(range=[0, 1.2], gridcolor="#2a3045", color="#64748b", tickformat=".0%"),
        yaxis=dict(color="#94a3b8", autorange="reversed"),
        margin=dict(t=10, b=10, l=10, r=60),
        height=max(140, len(threat_signals) * 36 + 40),
        font=dict(color="#94a3b8"),
    )
    return fig


def make_batch_donut(spam_count: int, ham_count: int) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=["🚨 Spam", "✅ Ham"],
        values=[spam_count, ham_count],
        hole=0.6,
        marker=dict(colors=["#ef4444", "#22c55e"], line=dict(color="#0f1117", width=2)),
        textfont=dict(color="#e2e8f0", size=13),
    ))
    fig.update_layout(
        paper_bgcolor="#0f1117",
        margin=dict(t=20, b=20, l=20, r=20),
        height=280,
        legend=dict(font=dict(color="#94a3b8")),
        annotations=[dict(text=f"<b>{spam_count+ham_count}</b><br>msgs", x=0.5, y=0.5, font_size=16, showarrow=False, font=dict(color="#e2e8f0"))],
    )
    return fig


def highlight_threats(text: str) -> str:
    """HTML-highlight common spam trigger phrases in the message preview."""
    HIGH_PATTERNS = [
        r'\b(URGENT|urgent|WINNER|winner|FREE|free|CLAIM|claim|CONGRATULATIONS|congratulations|SUSPENDED|suspended|LOCKED|locked|VERIFY|verify|COMPROMISED|compromised|GUARANTEED|guaranteed|PRIZE|prize|JACKPOT|jackpot)\b',
        r'\b(nigerian?|barrister|bitcoin|BTC|ETH|ethereum|viagra|cialis|pharmacy|paypal|apple id|wire transfer|seed phrase|private key)\b',
    ]
    SUS_PATTERNS = [
        r'https?://\S+',
        r'[\$£€¥₹]\s*\d[\d,\.]*',
        r'(?:transfer|send)\s+(?:funds?|money)',
    ]
    html = re.sub(r'<', '&lt;', re.sub(r'>', '&gt;', text))
    for pat in HIGH_PATTERNS:
        html = re.sub(pat, r'<span class="hl-spam">\1</span>', html, flags=re.IGNORECASE)
    for pat in SUS_PATTERNS:
        html = re.sub(pat, lambda m: f'<span class="hl-sus">{m.group()}</span>', html)
    return html


def text_forensics(text: str) -> dict:
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    urls = re.findall(r'https?://\S+|www\.\S+', text)
    money = re.findall(r'[\$£€¥₹]\s*\d[\d,\.]*', text)
    exclaims = text.count('!')
    letters = [c for c in text if c.isalpha()]
    caps_pct = (sum(1 for c in letters if c.isupper()) / len(letters) * 100) if letters else 0
    caps_words = len(re.findall(r'\b[A-Z]{3,}\b', text))
    return {
        "word_count": len(words),
        "char_count": len(text),
        "sentence_count": max(1, len([s for s in sentences if s.strip()])),
        "url_count": len(urls),
        "money_mentions": len(money),
        "exclamation_count": exclaims,
        "caps_ratio": round(caps_pct, 1),
        "caps_word_count": caps_words,
    }


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Sentinel Spam Shield")
    st.caption("Enterprise Email & SMS Threat Analyzer")
    st.divider()
    st.markdown("### ⚙️ Engine Info")
    st.markdown("""
- **ML Model**: Calibrated LinearSVC  
- **Features**: TF-IDF Bigrams, 6K vocab  
- **Heuristic Engine**: 25+ rule patterns  
- **Fusion**: Bayesian OR blending  
- **Training data**: 5,169 SMS + email spam augmentation
    """)
    st.divider()
    st.markdown("### 🔑 Threat Categories")
    for cat in ["🎣 Phishing / Account Takeover", "👑 419 Advance-Fee Fraud", "🪙 Crypto Scams", "💊 Pharmaceutical Spam", "💼 Employment Scams", "🏆 Lottery / Prize Scams", "📄 Business Email Compromise"]:
        st.markdown(f"&nbsp;&nbsp;{cat}", unsafe_allow_html=True)
    st.divider()
    st.caption("v2.0 · Built with Streamlit + scikit-learn + NLTK")

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("## 🛡️ Sentinel Spam Shield")
st.markdown("##### Enterprise-Grade AI-Powered Email & SMS Threat Detection")
st.divider()

# ── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔍 Single Message Inspector",
    "📂 Batch Scanner",
    "📊 Dataset Analytics",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Message Inspector
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1.1, 1], gap="large")

    with col_left:
        st.markdown("#### 📥 Message Input")

        preset_name = st.selectbox("⚡ Quick Load Preset", ["— Type your own message —"] + list(PRESETS.keys()))
        preset_text = PRESETS.get(preset_name, "")

        input_text = st.text_area(
            "Message Body",
            value=preset_text,
            height=200,
            placeholder="Paste or type any email or SMS message here…",
            label_visibility="collapsed",
        )

        scan_btn = st.button("🔍 Deep Threat Scan", use_container_width=True)

        # Text forensics panel (always shown when text is present)
        if input_text and input_text.strip():
            stats = text_forensics(input_text)
            st.markdown("---")
            st.markdown("#### 🧪 Text Forensics")
            fc1, fc2, fc3, fc4 = st.columns(4)
            fc1.metric("Words", stats["word_count"])
            fc2.metric("Characters", stats["char_count"])
            fc3.metric("URLs", stats["url_count"])
            fc4.metric("Money Refs", stats["money_mentions"])
            fc5, fc6, fc7, fc8 = st.columns(4)
            fc5.metric("Sentences", stats["sentence_count"])
            fc6.metric("Exclaims (!)", stats["exclamation_count"])
            fc7.metric("CAPS Ratio", f"{stats['caps_ratio']}%")
            fc8.metric("ALL-CAPS Words", stats["caps_word_count"])

    with col_right:
        if scan_btn or preset_name != "— Type your own message —":
            if not input_text or not input_text.strip():
                st.warning("⚠️ Please enter a message to scan.")
            else:
                try:
                    result = predict(input_text)
                    label = result["label"]
                    spam_prob = result["spam_prob"]
                    ml_prob = result["ml_spam_prob"]
                    h_score = result["heuristic_score"]
                    confidence = result["confidence"]
                    signals = result["threat_signals"]
                    is_spam = label == "spam"

                    # ── KPI Cards ────────────────────────────────────────
                    card_cls = "spam-card" if is_spam else "ham-card"
                    verdict_icon = "🚨" if is_spam else "✅"
                    verdict_text = "SPAM" if is_spam else "SAFE (HAM)"
                    verdict_color = "#ef4444" if is_spam else "#22c55e"

                    kc1, kc2, kc3 = st.columns(3)
                    kc1.markdown(f"""
                    <div class="metric-card {card_cls}">
                        <div class="metric-value" style="color:{verdict_color}">{verdict_icon} {verdict_text}</div>
                        <div class="metric-label">Verdict</div>
                    </div>""", unsafe_allow_html=True)
                    kc2.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="color:{verdict_color}">{confidence*100:.1f}%</div>
                        <div class="metric-label">Confidence</div>
                    </div>""", unsafe_allow_html=True)
                    kc3.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="color:#f59e0b">{len(signals)}</div>
                        <div class="metric-label">Threat Signals</div>
                    </div>""", unsafe_allow_html=True)

                    # ── Gauge ────────────────────────────────────────────
                    st.plotly_chart(make_gauge(spam_prob), use_container_width=True, config={"displayModeBar": False})

                    # ── Score Breakdown Bars ─────────────────────────────
                    st.markdown("##### 🔬 Score Breakdown")
                    st.plotly_chart(make_confidence_bars(ml_prob, h_score, spam_prob), use_container_width=True, config={"displayModeBar": False})

                except ValueError as exc:
                    st.warning(f"⚠️ {exc}")
                except Exception as exc:
                    st.error(f"❌ Unexpected error: {exc}")
        else:
            st.info("👈 Enter or load a message on the left, then click **🔍 Deep Threat Scan**")

    # ── Full-width analysis (shown after scan) ────────────────────────────
    if scan_btn or preset_name != "— Type your own message —":
        if input_text and input_text.strip():
            try:
                result = predict(input_text)
                label = result["label"]
                spam_prob = result["spam_prob"]
                ml_prob = result["ml_spam_prob"]
                h_score = result["heuristic_score"]
                signals = result["threat_signals"]
                is_spam = label == "spam"

                st.divider()
                r1, r2, r3 = st.columns([1.1, 1, 1], gap="large")

                with r1:
                    # ── Threat Radar ─────────────────────────────────────
                    st.markdown("##### 🕸️ Threat Category Radar")
                    st.plotly_chart(make_radar(signals, h_score), use_container_width=True, config={"displayModeBar": False})

                with r2:
                    # ── Triggered Signals ────────────────────────────────
                    st.markdown("##### 🚩 Triggered Threat Signals")
                    if signals:
                        for sig in signals:
                            w = {
                                "419 Advance-Fee Fraud": "high", "Account Threat": "high",
                                "Brand Phishing": "high", "Credential Phishing": "high",
                                "Payment Method": "high", "Pharmaceutical Spam": "high",
                                "Crypto Wallet Threat": "high", "Crypto Doubling Scam": "high",
                                "IP-Based URL": "high", "Homograph Domain": "high",
                                "419 Royalty Scam": "high", "Bank Account Threat": "high",
                            }.get(sig, "med")
                            cls = f"signal-{w}"
                            st.markdown(f'<span class="signal-pill {cls}">{"🔴" if w=="high" else "🟡"} {sig}</span>', unsafe_allow_html=True)
                        st.markdown("")
                        fig_sig = make_signal_hbar(signals)
                        if fig_sig:
                            st.plotly_chart(fig_sig, use_container_width=True, config={"displayModeBar": False})
                    else:
                        st.success("✅ No suspicious patterns detected")

                with r3:
                    # ── Highlighted Message Preview ───────────────────────
                    st.markdown("##### 🔍 Highlighted Message Preview")
                    highlighted = highlight_threats(input_text)
                    preview = textwrap.shorten(highlighted, width=600, placeholder="…")
                    st.markdown(f"""
                    <div style="background:#1e2535;border:1px solid #2a3045;border-radius:10px;padding:14px;
                                font-size:0.85rem;line-height:1.6;color:#94a3b8;max-height:280px;overflow-y:auto;">
                        {highlighted}
                    </div>
                    <p style="font-size:0.72rem;color:#4a5568;margin-top:6px;">
                        🔴 High-risk keywords &nbsp; 🟡 Suspicious patterns
                    </p>
                    """, unsafe_allow_html=True)

            except Exception:
                pass

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch Scanner
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### 📂 Batch Message Scanner")
    st.markdown("Upload a CSV file with a column containing messages, or paste multiple messages (one per line) below.")

    bc1, bc2 = st.columns([1, 1], gap="large")

    with bc1:
        upload_method = st.radio("Input Method", ["📋 Paste Messages", "📄 Upload CSV"], horizontal=True)

        messages_to_scan = []

        if upload_method == "📋 Paste Messages":
            bulk_text = st.text_area("Paste messages (one per line)", height=200)
            if bulk_text.strip():
                messages_to_scan = [line.strip() for line in bulk_text.splitlines() if line.strip()]
        else:
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
            if uploaded_file:
                df_up = pd.read_csv(uploaded_file)
                col_choice = st.selectbox("Select the message column", df_up.columns.tolist())
                if col_choice:
                    messages_to_scan = df_up[col_choice].dropna().astype(str).tolist()

        if messages_to_scan:
            st.info(f"📨 {len(messages_to_scan)} messages loaded. Click the button to scan.")

        scan_batch_btn = st.button("🚀 Scan All Messages", use_container_width=True, disabled=not messages_to_scan)

    with bc2:
        if scan_batch_btn and messages_to_scan:
            results_list = []
            progress = st.progress(0)
            for i, msg in enumerate(messages_to_scan[:500]):  # cap at 500
                try:
                    r = predict(msg)
                    results_list.append({
                        "message": msg[:120],
                        "label": r["label"],
                        "confidence": r["confidence"],
                        "spam_prob": r["spam_prob"],
                        "ml_spam_prob": r["ml_spam_prob"],
                        "heuristic_score": r["heuristic_score"],
                        "threat_signals": ", ".join(r["threat_signals"]),
                    })
                except Exception:
                    results_list.append({"message": msg[:120], "label": "error", "confidence": 0, "spam_prob": 0, "ml_spam_prob": 0, "heuristic_score": 0, "threat_signals": ""})
                progress.progress((i + 1) / min(len(messages_to_scan), 500))

            results_df = pd.DataFrame(results_list)
            st.session_state["batch_results"] = results_df
            st.success(f"✅ Scanned {len(results_df)} messages.")

    # Show batch results
    if "batch_results" in st.session_state:
        df_res = st.session_state["batch_results"]
        spam_n = (df_res["label"] == "spam").sum()
        ham_n  = (df_res["label"] == "ham").sum()

        st.divider()
        br1, br2, br3, br4 = st.columns(4)
        br1.metric("Total Scanned", len(df_res))
        br2.metric("🚨 Spam", spam_n)
        br3.metric("✅ Ham", ham_n)
        br4.metric("Spam Rate", f"{spam_n/len(df_res)*100:.1f}%")

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("##### Distribution")
            st.plotly_chart(make_batch_donut(spam_n, ham_n), use_container_width=True, config={"displayModeBar": False})
        with chart_col2:
            st.markdown("##### Confidence Distribution")
            fig_hist = px.histogram(
                df_res, x="spam_prob", nbins=20,
                color_discrete_sequence=["#3b82f6"],
                labels={"spam_prob": "Spam Probability"},
            )
            fig_hist.update_layout(
                paper_bgcolor="#0f1117", plot_bgcolor="#1e2535",
                xaxis=dict(gridcolor="#2a3045", color="#64748b"),
                yaxis=dict(gridcolor="#2a3045", color="#64748b"),
                font=dict(color="#94a3b8"),
                margin=dict(t=20, b=30, l=20, r=20),
                height=280,
            )
            st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

        st.markdown("##### 📋 Detailed Results")
        display_df = df_res.copy()
        display_df["spam_prob"] = (display_df["spam_prob"] * 100).round(1).astype(str) + "%"
        display_df["confidence"] = (display_df["confidence"] * 100).round(1).astype(str) + "%"
        st.dataframe(
            display_df[["message", "label", "confidence", "spam_prob", "threat_signals"]],
            use_container_width=True,
            hide_index=True,
        )

        # Download
        csv_bytes = df_res.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Results as CSV",
            data=csv_bytes,
            file_name="spam_scan_results.csv",
            mime="text/csv",
        )

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Dataset Analytics
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### 📊 Training Dataset Analytics")

    @st.cache_data
    def load_dataset_stats():
        df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
        df.columns = ["label", "text"]
        df = df.drop_duplicates(keep="first").reset_index(drop=True)
        df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))
        df["char_count"] = df["text"].apply(len)
        return df

    try:
        ds = load_dataset_stats()
        spam_df = ds[ds["label"] == "spam"]
        ham_df  = ds[ds["label"] == "ham"]

        da1, da2, da3, da4 = st.columns(4)
        da1.metric("Total Messages", len(ds))
        da2.metric("Spam Messages", len(spam_df))
        da3.metric("Ham Messages", len(ham_df))
        da4.metric("Spam Rate", f"{len(spam_df)/len(ds)*100:.1f}%")

        st.divider()
        dc1, dc2 = st.columns(2)

        with dc1:
            st.markdown("##### Class Distribution")
            fig_pie = go.Figure(go.Pie(
                labels=["🚨 Spam", "✅ Ham"],
                values=[len(spam_df), len(ham_df)],
                hole=0.55,
                marker=dict(colors=["#ef4444", "#22c55e"], line=dict(color="#0f1117", width=2)),
            ))
            fig_pie.update_layout(paper_bgcolor="#0f1117", margin=dict(t=20,b=20,l=20,r=20), height=280, legend=dict(font=dict(color="#94a3b8")))
            st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

        with dc2:
            st.markdown("##### Message Length Distribution")
            fig_len = go.Figure()
            fig_len.add_trace(go.Histogram(x=spam_df["word_count"], name="Spam", nbinsx=30, marker_color="#ef4444", opacity=0.7))
            fig_len.add_trace(go.Histogram(x=ham_df["word_count"].sample(min(len(spam_df)*3, len(ham_df))), name="Ham", nbinsx=30, marker_color="#22c55e", opacity=0.7))
            fig_len.update_layout(
                barmode="overlay",
                paper_bgcolor="#0f1117", plot_bgcolor="#1e2535",
                xaxis=dict(title="Word Count", gridcolor="#2a3045", color="#64748b"),
                yaxis=dict(title="Count", gridcolor="#2a3045", color="#64748b"),
                font=dict(color="#94a3b8"),
                legend=dict(font=dict(color="#94a3b8")),
                margin=dict(t=20, b=30, l=20, r=20),
                height=280,
            )
            st.plotly_chart(fig_len, use_container_width=True, config={"displayModeBar": False})

        st.divider()
        dc3, dc4 = st.columns(2)

        with dc3:
            st.markdown("##### Top Spam Keywords")
            spam_words = " ".join(spam_df["text"].fillna("").str.lower()).split()
            common_words = {"to", "a", "the", "you", "your", "and", "in", "is", "for", "of", "on", "i", "it", "or", "are", "this", "have", "be", "with", "my", "that", "can", "will", "s", "u", "at", "from", "has", "our", "if", "us", "we", "do", "not", "now", "so", "as", "an", "by", "get", "me"}
            spam_words = [w for w in spam_words if len(w) > 2 and w not in common_words]
            top_spam = Counter(spam_words).most_common(15)
            words_s, counts_s = zip(*top_spam)
            fig_sw = go.Figure(go.Bar(
                x=counts_s, y=words_s, orientation="h",
                marker=dict(color="#ef4444", opacity=0.85),
            ))
            fig_sw.update_layout(
                paper_bgcolor="#0f1117", plot_bgcolor="#1e2535",
                xaxis=dict(gridcolor="#2a3045", color="#64748b"),
                yaxis=dict(color="#94a3b8", autorange="reversed"),
                margin=dict(t=10, b=10, l=10, r=20),
                height=360, font=dict(color="#94a3b8"),
            )
            st.plotly_chart(fig_sw, use_container_width=True, config={"displayModeBar": False})

        with dc4:
            st.markdown("##### Top Ham Keywords")
            ham_words = " ".join(ham_df["text"].fillna("").str.lower().sample(min(1000, len(ham_df)))).split()
            ham_words = [w for w in ham_words if len(w) > 2 and w not in common_words]
            top_ham = Counter(ham_words).most_common(15)
            words_h, counts_h = zip(*top_ham)
            fig_hw = go.Figure(go.Bar(
                x=counts_h, y=words_h, orientation="h",
                marker=dict(color="#22c55e", opacity=0.85),
            ))
            fig_hw.update_layout(
                paper_bgcolor="#0f1117", plot_bgcolor="#1e2535",
                xaxis=dict(gridcolor="#2a3045", color="#64748b"),
                yaxis=dict(color="#94a3b8", autorange="reversed"),
                margin=dict(t=10, b=10, l=10, r=20),
                height=360, font=dict(color="#94a3b8"),
            )
            st.plotly_chart(fig_hw, use_container_width=True, config={"displayModeBar": False})

        st.divider()
        st.markdown("##### 📈 Model Performance Metrics (Training Result)")
        perf_data = {
            "Model": ["ComplementNB", "Logistic Regression", "LinearSVC (Best ✅)"],
            "Accuracy": ["95.8%", "96.9%", "98.0%"],
            "Precision (Spam)": ["78.3%", "87.3%", "93.9%"],
            "Recall (Spam)": ["94.2%", "89.9%", "90.6%"],
            "F1 (Spam)": ["85.5%", "88.6%", "92.3%"],
        }
        st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)

    except FileNotFoundError:
        st.warning("⚠️ spam.csv not found in the working directory. Run from the project root.")
