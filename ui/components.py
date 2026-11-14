from html import escape

import plotly.graph_objects as go
import streamlit as st

import train
from data_preprocessing import transform_text
from spam_model import DEFAULT_THRESHOLD, SpamDetector

from .theme import AMBER, BLUE, BORDER, CARD_2, GREEN, MUTED, RED, TEXT, VERDICT_COLORS

REPO_URL = "https://github.com/Oceangupta01/email_spam_detection"


@st.cache_resource(show_spinner="Loading the spam model (first run may train it, ~20 s)...")
def get_detector() -> SpamDetector:
    return SpamDetector()


@st.cache_data(show_spinner="Loading datasets...")
def get_datasets():
    df = train.load_datasets()
    _, test_df = train.split(df)
    test_idx = test_df.index
    df["clean"] = df.text.map(transform_text)
    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"
    df["label_name"] = df.label.map({0: "Ham", 1: "Spam"})
    df["characters"] = df.text.str.len()
    df["words"] = df.text.str.split().str.len()
    return df


def threshold() -> float:
    return st.session_state.get("threshold", DEFAULT_THRESHOLD)


def sidebar(detector: SpamDetector):
    m = detector.metrics
    with st.sidebar:
        st.markdown("### Detection settings")
        st.session_state.setdefault("threshold", DEFAULT_THRESHOLD)
        st.slider(
            "Spam threshold", 0.05, 0.95, step=0.05, key="threshold",
            help="Messages whose spam probability is at or above this value are labelled Spam. "
                 "Lower it to catch more spam (more false alarms); raise it to be stricter.",
        )
        sweep = {round(r["threshold"], 2): r for r in m["threshold_sweep"]}
        row = sweep.get(round(threshold(), 2))
        if row:
            st.caption(f"At this threshold on the test set: precision **{row['precision']:.1%}**, "
                       f"recall **{row['recall']:.1%}**, F1 **{row['f1']:.1%}**.")
        st.divider()
        st.markdown("### Model")
        st.markdown(
            f"- **Algorithm:** Calibrated Linear SVM\n"
            f"- **Features:** TF-IDF words + word pairs ({m['vocabulary_size']:,})\n"
            f"- **Trained on:** {m['data']['train']:,} SMS + email messages\n"
            f"- **Test accuracy:** {m['test']['accuracy']:.1%}\n"
            f"- **Email spam F1:** {m['test_by_source']['Email']['f1']:.1%}\n"
            f"- **Version:** v{m['model_version']} · {m['trained_at'][:10]}"
        )
        st.divider()
        st.markdown(f"**Author:** Ocean Gupta  \n**Institution:** GCET, Jammu  \n[Source on GitHub]({REPO_URL})")


def hero(title: str, subtitle: str, badges=()):
    chips = "".join(f'<span class="badge">{escape(b)}</span>' for b in badges)
    st.markdown(
        f'<div class="hero"><h1>{escape(title)}</h1><p>{escape(subtitle)}</p>'
        f'<div class="badges">{chips}</div></div>',
        unsafe_allow_html=True,
    )


def kpi(label: str, value: str, sub: str = "", accent: str = BLUE, delta: float | None = None, delta_fmt="{:+.1%}"):
    delta_html = ""
    if delta is not None:
        cls = "up" if delta >= 0 else "down"
        delta_html = f'<span class="{cls}">{delta_fmt.format(delta)}</span> '
    st.markdown(
        f'<div class="kpi" style="--accent:{accent}"><div class="label">{escape(label)}</div>'
        f'<div class="value">{escape(value)}</div><div class="sub">{delta_html}{escape(sub)}</div></div>',
        unsafe_allow_html=True,
    )


def kpi_row(items):
    for col, item in zip(st.columns(len(items)), items):
        with col:
            kpi(**item)


def section(title: str, subtitle: str = ""):
    st.markdown(f'<div class="section-title"><h3>{escape(title)}</h3></div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{escape(subtitle)}</div>', unsafe_allow_html=True)


def info_card(title: str, body_html: str):
    st.markdown(f'<div class="info-card"><h4>{escape(title)}</h4>{body_html}</div>', unsafe_allow_html=True)


def steps_html(steps) -> str:
    return "".join(
        f'<div class="step"><div class="n">{i}</div><div><div class="t">{escape(t)}</div>'
        f'<div class="d">{escape(d)}</div></div></div>'
        for i, (t, d) in enumerate(steps, 1)
    )


VERDICT_TEXT = {
    "Spam": "This message matches the patterns of unsolicited or fraudulent mail. Don't click links, reply, or share details.",
    "Suspicious": "The model's text score is below the threshold, but the rule layer found a phishing or bulk-spam "
                  "pattern (credential requests with links or pressure, or several strong spam signals). "
                  "Verify the sender independently.",
    "Not Spam": "This message looks like legitimate personal or business communication.",
}


def verdict_card(analysis):
    color = VERDICT_COLORS[analysis.verdict]
    st.markdown(
        f'<div class="verdict" style="--accent:{color}"><div class="tag">Verdict · {analysis.risk_level} risk</div>'
        f'<div class="title">{analysis.verdict}</div>'
        f'<div class="desc">{escape(VERDICT_TEXT[analysis.verdict])}</div></div>',
        unsafe_allow_html=True,
    )


def gauge(value: float, title: str, threshold_value: float | None = None, suffix="%", height=230):
    pct = value * 100 if suffix == "%" else value
    color = RED if pct >= 75 else AMBER if pct >= 40 else GREEN
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=pct,
        number={"suffix": suffix, "font": {"size": 40, "color": TEXT}, "valueformat": ".1f" if suffix == "%" else ".0f"},
        title={"text": title, "font": {"size": 14, "color": MUTED}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": MUTED, "tickfont": {"color": MUTED}},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": CARD_2, "borderwidth": 0,
            "steps": [{"range": [0, 40], "color": "rgba(166,227,161,0.12)"},
                      {"range": [40, 75], "color": "rgba(249,226,175,0.12)"},
                      {"range": [75, 100], "color": "rgba(243,139,168,0.14)"}],
            "threshold": {"line": {"color": TEXT, "width": 3}, "thickness": 0.85,
                          "value": threshold_value * 100} if threshold_value is not None else None,
        },
    ))
    fig.update_layout(height=height, margin=dict(l=25, r=25, t=50, b=5))
    return fig


def _signal_html(s) -> str:
    chips = "".join(f'<span class="chip">{escape(str(m))[:60]}</span>' for m in s.matches)
    value = f"{s.value:g}%" if s.key == "caps" else f"{s.value:g} found"
    return (f'<div class="signal {"on" if s.triggered else ""}"><div class="head"><span>{escape(s.label)}</span>'
            f'<span class="pts">{"+" + str(int(s.weight)) + " risk" if s.triggered else value}</span></div>'
            f'<div class="desc">{escape(s.description)}</div>{chips}</div>')


def signal_cards(signals):
    on = sorted((s for s in signals if s.triggered), key=lambda s: -s.weight)
    off = [s for s in signals if not s.triggered]
    if on:
        st.markdown("".join(_signal_html(s) for s in on), unsafe_allow_html=True)
    else:
        st.success("No risk signals triggered: no links, pressure tactics, prizes or credential requests found.")
    if off:
        with st.expander(f"Signals not triggered ({len(off)})"):
            st.markdown("".join(_signal_html(s) for s in off), unsafe_allow_html=True)


def highlight_html(chunks) -> str:
    peak = max((abs(w) for _, w in chunks), default=0) or 1.0
    parts = []
    for chunk, w in chunks:
        safe = escape(chunk)
        if abs(w) < 0.05 * peak:
            parts.append(safe)
            continue
        alpha = 0.15 + 0.6 * min(1.0, abs(w) / peak)
        rgb = "243,139,168" if w > 0 else "166,227,161"
        parts.append(f'<span style="background:rgba({rgb},{alpha:.2f})" title="weight {w:+.3f}">{safe}</span>')
    return f'<div class="highlight-box">{"".join(parts)}</div>'


def footer():
    st.markdown(
        f'<div class="app-footer">Email Spam Detection System · Ocean Gupta, GCET Jammu · '
        f'Calibrated Linear SVM + TF-IDF · scikit-learn · Streamlit · Plotly · '
        f'<a href="{REPO_URL}" target="_blank">GitHub</a> · MIT License</div>',
        unsafe_allow_html=True,
    )


def empty_state(title: str, body: str):
    st.markdown(
        f'<div class="info-card" style="text-align:center;padding:2.2rem;border-style:dashed;border-color:{BORDER}">'
        f'<h4>{escape(title)}</h4><p>{escape(body)}</p></div>',
        unsafe_allow_html=True,
    )
