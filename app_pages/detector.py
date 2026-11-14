from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ui.components import (
    empty_state, gauge, get_datasets, get_detector, hero, highlight_html, info_card, kpi_row, section,
    signal_cards, steps_html, threshold, verdict_card,
)
from ui.theme import AMBER, BLUE, CLASS_COLORS, GREEN, MUTED, RED, VERDICT_COLORS

SAMPLES = {
    "Prize scam": "Congratulations! You have been selected as the WINNER of a $1,000 Walmart gift card. "
                  "Click here http://bit.ly/claim-prize to claim your reward now. Offer expires in 24 hours!",
    "Phishing": "Your Amazon order #112-4471 could not be delivered. Please update your payment information "
                "here to avoid cancellation of your order.",
    "Bank alert": "URGENT: Your account has been suspended due to unusual activity. Verify your bank details "
                  "immediately at www.secure-login-verify.com or your card will be blocked.",
    "Promo email": "Save up to 70% on Life Insurance. Why spend more than you have to? Life Quote Savings gives "
                   "you FREE access to the very best companies and the lowest rates. Get your free quote today!",
    "SMS spam": "FREE entry! Win a brand new iPhone 15. Text WIN to 80086 now. T&Cs apply. 18+ only.",
    "Work email": "Hi Priya, can we move tomorrow's project review to 3pm? I've attached the updated slides and "
                  "the test results. Let me know if that works. Thanks, Rahul",
    "Personal": "Hey! Are we still on for dinner tonight? I'll pick you up around 8. Mom says hi.",
}

detector = get_detector()
m = detector.metrics
st.session_state.setdefault("history", [])
st.session_state.setdefault("analyzed_text", "")


def _use_sample(name):
    st.session_state.message = SAMPLES[name]


def _clear():
    st.session_state.message = ""
    st.session_state.analyzed_text = ""


def _analyze():
    text = st.session_state.get("message", "").strip()
    st.session_state.empty_input = not text
    if not text:
        return
    st.session_state.analyzed_text = text
    a = detector.analyze(text, threshold())
    st.session_state.history.append({
        "time": datetime.now().strftime("%H:%M:%S"), "preview": a.text[:80].replace("\n", " "),
        "verdict": a.verdict, "spam_probability": a.spam_probability, "risk_score": a.risk_score,
        "words": a.stats["words"],
    })


hero(
    "Email Spam Detector",
    "Paste any email or SMS to get an instant verdict with a calibrated spam probability, a word-by-word "
    "explanation and phishing risk signals.",
    badges=[f"{m['test']['accuracy']:.1%} test accuracy", f"ROC-AUC {m['test']['roc_auc']:.3f}",
            f"{m['data']['total']:,} labelled messages", "SMS + Email", "Explainable AI"],
)

base_email = m["baseline"]["by_source"]["Email"]["recall"]
new_email = m["test_by_source"]["Email"]["recall"]
kpi_row([
    dict(label="Test accuracy", value=f"{m['test']['accuracy']:.1%}", sub="on held-out messages", accent=BLUE),
    dict(label="Email spam caught", value=f"{new_email:.1%}", sub="vs original model", delta=new_email - base_email, accent=GREEN),
    dict(label="Precision", value=f"{m['test']['precision']:.1%}", sub="of Spam verdicts are correct", accent=AMBER),
    dict(label="Messages analyzed", value=f"{len(st.session_state.history)}", sub="this session", accent=RED),
])

left, right = st.columns([3, 2], gap="large")
with left:
    section("Analyze a message", "Try a sample or paste your own text. Raw emails with headers or HTML are supported.")
    sample_cols = st.columns(4)
    for i, name in enumerate(SAMPLES):
        sample_cols[i % 4].button(name, key=f"sample_{name}", on_click=_use_sample, args=(name,), width="stretch")
    st.text_area("Message", key="message", height=210, label_visibility="collapsed",
                 placeholder="Paste or type an email / SMS message here...")
    b1, b2, _ = st.columns([1.3, 1, 2])
    b1.button("Analyze message", type="primary", icon=":material/search:", on_click=_analyze, width="stretch")
    b2.button("Clear", icon=":material/close:", on_click=_clear, width="stretch")
    if st.session_state.pop("empty_input", False):
        st.warning("Please enter a message before analyzing.")

with right:
    section("How the detector decides")
    info_card("Three layers, one verdict", steps_html([
        ("Normalize", "Links, money amounts, phone numbers, percentages, ALL-CAPS and '!' become tokens."),
        ("Score", "A calibrated Linear SVM over 1-2 word TF-IDF features outputs a spam probability."),
        ("Inspect", "Rule-based signals flag phishing (credential requests + pressure or links) and bulk-spam topics."),
        ("Decide", f"Spam if probability >= {threshold():.0%}; Suspicious if phishing or strong spam rules fire."),
    ]))

if st.session_state.analyzed_text:
    a = detector.analyze(st.session_state.analyzed_text, threshold())
    section("Result")
    c1, c2, c3 = st.columns([1.6, 1, 1], gap="medium")
    with c1:
        verdict_card(a)
        st.caption(f"Spam probability {a.spam_probability:.1%} · threshold {a.threshold:.0%} · "
                   f"rule-layer alert {'raised' if a.phishing_like else 'not raised'}")
    c2.plotly_chart(gauge(a.spam_probability, "Spam probability (model)", a.threshold))
    c3.plotly_chart(gauge(a.risk_score, "Risk-signal score", suffix=" / 100"))

    kpi_row([
        dict(label="Characters", value=f"{a.stats['characters']:,}", accent=BLUE),
        dict(label="Words", value=f"{a.stats['words']:,}", sub=f"avg {a.stats['avg_word_length']} letters", accent=BLUE),
        dict(label="Model features matched", value=f"{a.stats['known_terms']}", sub="known words & word pairs", accent=AMBER),
        dict(label="Signals triggered", value=f"{sum(s.triggered for s in a.signals)} / {len(a.signals)}",
             sub=f"risk level {a.risk_level}", accent=VERDICT_COLORS[a.verdict]),
    ])

    tab1, tab2, tab3 = st.tabs(["Why this verdict", "Risk signals", "Message breakdown"])
    with tab1:
        t1, t2 = st.columns([1, 1.2], gap="large")
        with t1:
            st.markdown("**Top features influencing the score**")
            if a.contributions:
                cdf = pd.DataFrame(a.contributions).iloc[::-1]
                cdf["direction"] = cdf.contribution.map(lambda v: "Pushes to Spam" if v > 0 else "Pushes to Ham")
                fig = px.bar(cdf, x="contribution", y="term", orientation="h", color="direction",
                             color_discrete_map={"Pushes to Spam": RED, "Pushes to Ham": GREEN})
                fig.update_layout(height=420, yaxis_title=None, xaxis_title="Contribution to spam score", legend_title=None)
                st.plotly_chart(fig)
            else:
                empty_state("No known features", "None of the words in this message appear in the model's vocabulary.")
        with t2:
            st.markdown("**Word-level highlighting**")
            st.markdown(highlight_html(a.highlights), unsafe_allow_html=True)
            st.markdown(
                f'<div class="legend"><span style="background:rgba(243,139,168,.55);padding:0 6px;border-radius:4px">red</span> '
                f'words push towards spam · <span style="background:rgba(166,227,161,.55);padding:0 6px;border-radius:4px">green</span> '
                f'words push towards ham · hover a word for its weight</div>', unsafe_allow_html=True)
    with tab2:
        r1, r2 = st.columns([1.1, 1], gap="large")
        with r1:
            st.markdown("**Signal profile**")
            sig = pd.DataFrame([{"signal": s.label, "weight": s.weight} for s in a.signals])
            fig = go.Figure(go.Barpolar(r=sig.weight.clip(lower=0.5), theta=sig.signal,
                                        marker_color=[RED if w > 0 else "rgba(166,173,200,0.25)" for w in sig.weight],
                                        hovertemplate="%{theta}: %{r:.0f} pts<extra></extra>"))
            fig.update_layout(height=430, margin=dict(l=90, r=90, t=40, b=40),
                              polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(showticklabels=False, gridcolor="rgba(166,173,200,0.15)"),
                                         angularaxis=dict(gridcolor="rgba(166,173,200,0.15)", tickfont=dict(size=11))))
            st.plotly_chart(fig)
        with r2:
            st.markdown("**Signal details**")
            signal_cards(a.signals)
    with tab3:
        d1, d2 = st.columns(2, gap="large")
        with d1:
            st.markdown("**Class probabilities**")
            fig = go.Figure(go.Pie(labels=["Spam", "Ham"], values=[a.spam_probability, 1 - a.spam_probability], hole=0.62,
                                   marker_colors=[RED, GREEN], sort=False, textinfo="percent"))
            fig.update_layout(height=320, annotations=[dict(text=a.verdict, showarrow=False, font=dict(size=20))])
            st.plotly_chart(fig)
        with d2:
            st.markdown("**Compared with the training data (median length)**")
            ds = get_datasets()
            med = ds.groupby("label_name")[["characters", "words"]].median()
            comp = pd.DataFrame({
                "group": ["This message", "Typical spam", "Typical ham"],
                "words": [a.stats["words"], med.loc["Spam", "words"], med.loc["Ham", "words"]],
            })
            fig = px.bar(comp, x="group", y="words", color="group", text_auto=".0f",
                         color_discrete_map={"This message": BLUE, "Typical spam": RED, "Typical ham": GREEN})
            fig.update_layout(height=320, showlegend=False, xaxis_title=None, yaxis_title="Words")
            st.plotly_chart(fig)

section("Session history", "Every message you analyze in this session is tracked here.")
hist = pd.DataFrame(st.session_state.history)
if hist.empty:
    empty_state("No messages analyzed yet", "Pick a sample above or paste your own message to start building your session dashboard.")
else:
    h1, h2 = st.columns([1, 2], gap="large")
    with h1:
        counts = hist.verdict.value_counts()
        fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.6,
                               marker_colors=[VERDICT_COLORS[v] for v in counts.index]))
        fig.update_layout(height=300, title="Verdict mix", annotations=[dict(text=f"{len(hist)}<br>checked", showarrow=False, font=dict(size=16))])
        st.plotly_chart(fig)
    with h2:
        hist["#"] = range(1, len(hist) + 1)
        fig = px.line(hist, x="#", y="spam_probability", markers=True, hover_data=["preview", "verdict"], title="Spam probability over time",
                      labels={"#": "Message #", "spam_probability": "Spam probability"})
        fig.update_traces(line_color=BLUE, marker=dict(size=10, color=[VERDICT_COLORS[v] for v in hist.verdict]))
        fig.add_hline(y=threshold(), line_dash="dash", line_color=MUTED, annotation_text="threshold")
        fig.update_layout(height=300, yaxis=dict(range=[0, 1.02], tickformat=".0%"), xaxis=dict(dtick=1))
        st.plotly_chart(fig)
    st.dataframe(
        hist.drop(columns="#").iloc[::-1], hide_index=True,
        column_config={
            "spam_probability": st.column_config.ProgressColumn("Spam probability", format="percent", min_value=0, max_value=1),
            "risk_score": st.column_config.NumberColumn("Risk score"),
            "preview": st.column_config.TextColumn("Message", width="large"),
        },
    )
    if st.button("Clear history", icon=":material/delete:"):
        st.session_state.history = []
        st.rerun()
