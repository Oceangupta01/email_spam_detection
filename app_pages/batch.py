import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix

from ui.components import empty_state, get_datasets, get_detector, hero, info_card, kpi_row, section, steps_html, threshold
from ui.theme import AMBER, BLUE, GREEN, MUTED, RED, VERDICT_COLORS

MAX_ROWS = 5000
TRUTHY = {"spam", "1", "true", "yes", "junk"}

detector = get_detector()

hero(
    "Batch Analysis",
    "Upload a CSV or text file of messages and screen them all at once: get verdicts, spam probabilities, "
    "risk signals and downloadable results, plus accuracy if your file already has labels.",
    badges=["CSV & TXT", f"Up to {MAX_ROWS:,} messages", "Optional ground-truth evaluation", "Export results"],
)


def _read_upload(file) -> pd.DataFrame:
    raw = file.getvalue()
    if file.name.lower().endswith(".txt"):
        text = raw.decode("utf-8", errors="replace")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return pd.DataFrame({"message": lines})
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode the file.")


def _guess(cols, keywords):
    for c in cols:
        if any(k in str(c).lower() for k in keywords):
            return c
    return None


up_col, help_col = st.columns([3, 2], gap="large")
with up_col:
    section("1. Choose your data")
    file = st.file_uploader("Upload messages", type=["csv", "txt"], label_visibility="collapsed",
                            help="CSV with one message per row, or TXT with one message per line.")
    demo = st.button("Use a demo batch (300 labelled messages the model never saw)", icon=":material/science:")
with help_col:
    section("How it works")
    info_card("Batch workflow", steps_html([
        ("Upload", "CSV (pick the text column) or TXT (one message per line)."),
        ("Optional labels", "Pick a label column (spam/ham or 1/0) to measure accuracy on your own data."),
        ("Analyze", "Every row is scored by the model and checked for phishing signals."),
        ("Explore & export", "Filter, sort and download the results as CSV."),
    ]))

if demo:
    ds = get_datasets()
    sample = ds[ds.split == "test"].groupby(["source", "label"]).sample(n=75, random_state=7)
    st.session_state.batch_input = pd.DataFrame({"message": sample.text.values, "label": sample.label_name.values,
                                                 "source": sample.source.values})
    st.session_state.batch_name = "demo batch"
    st.session_state.pop("batch_result", None)
elif file is not None and st.session_state.get("batch_name") != file.name:
    try:
        st.session_state.batch_input = _read_upload(file)
        st.session_state.batch_name = file.name
        st.session_state.pop("batch_result", None)
    except (ValueError, pd.errors.ParserError) as exc:
        st.error(f"Could not read **{file.name}**: {exc}")

data = st.session_state.get("batch_input")
if data is None or data.empty:
    empty_state("No data yet", "Upload a CSV / TXT file or load the demo batch to see a full screening report.")
    st.stop()

section("2. Configure columns", f"Loaded **{len(data):,}** rows from {st.session_state.batch_name}.")
cols = list(data.columns)
c1, c2, c3 = st.columns([2, 2, 1])
text_col = c1.selectbox("Message column", cols, index=cols.index(_guess(cols, ["message", "text", "body", "content", "v2"]) or cols[0]))
label_options = ["(none)"] + [c for c in cols if c != text_col]
guess_label = _guess([c for c in cols if c != text_col], ["label", "class", "category", "spam", "v1", "target"])
label_col = c2.selectbox("Ground-truth label column (optional)", label_options,
                         index=label_options.index(guess_label) if guess_label else 0)
c3.markdown("<br>", unsafe_allow_html=True)
go_btn = c3.button("Run analysis", type="primary", icon=":material/play_arrow:", width="stretch")

if go_btn:
    texts = data[text_col].fillna("").astype(str).head(MAX_ROWS).tolist()
    bar = st.progress(0.0, text="Analyzing messages...")
    result = detector.analyze_batch(texts, threshold(), progress=lambda p: bar.progress(p, text=f"Analyzing messages... {p:.0%}"))
    bar.empty()
    if label_col != "(none)":
        result["actual"] = data[label_col].head(MAX_ROWS).astype(str).str.strip().str.lower().isin(TRUTHY).map({True: "Spam", False: "Ham"}).values
    if "source" in data.columns and text_col != "source":
        result["source"] = data["source"].head(MAX_ROWS).values
    st.session_state.batch_result = result
    if len(data) > MAX_ROWS:
        st.info(f"Only the first {MAX_ROWS:,} rows were analyzed.")

res = st.session_state.get("batch_result")
if res is None:
    empty_state("Ready to analyze", "Choose the message column and press Run analysis.")
    st.stop()

res = res.copy()
t = threshold()
res["verdict"] = np.where(res.spam_probability >= t, "Spam", np.where(res.phishing_pattern, "Suspicious", "Not Spam"))

section("3. Screening report", f"Verdicts use the current threshold of {t:.0%} (change it in the sidebar; results update instantly).")
n = len(res)
counts = res.verdict.value_counts()
kpi_row([
    dict(label="Messages", value=f"{n:,}", sub="analyzed", accent=BLUE),
    dict(label="Spam", value=f"{counts.get('Spam', 0):,}", sub=f"{counts.get('Spam', 0) / n:.1%} of batch", accent=RED),
    dict(label="Suspicious", value=f"{counts.get('Suspicious', 0):,}", sub="phishing patterns", accent=AMBER),
    dict(label="Not spam", value=f"{counts.get('Not Spam', 0):,}", sub=f"{counts.get('Not Spam', 0) / n:.1%} of batch", accent=GREEN),
    dict(label="Mean spam probability", value=f"{res.spam_probability.mean():.1%}", sub=f"median risk score {res.risk_score.median():.0f}", accent=BLUE),
])

g1, g2 = st.columns([1, 1.6], gap="large")
with g1:
    fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.6, marker_colors=[VERDICT_COLORS[v] for v in counts.index]))
    fig.update_layout(height=340, title="Verdict distribution", annotations=[dict(text=f"{n:,}<br>messages", showarrow=False, font=dict(size=16))])
    st.plotly_chart(fig)
with g2:
    fig = px.histogram(res, x="spam_probability", color="verdict", nbins=40, color_discrete_map=VERDICT_COLORS,
                       title="Spam probability distribution")
    fig.add_vline(x=t, line_dash="dash", line_color=MUTED, annotation_text="threshold")
    fig.update_layout(height=340, bargap=0.05, xaxis_tickformat=".0%", yaxis_title="Messages", legend_title=None)
    st.plotly_chart(fig)

g3, g4 = st.columns(2, gap="large")
with g3:
    sig = res.signals.str.split(", ").explode().replace("", np.nan).dropna().value_counts().head(10).iloc[::-1]
    if sig.empty:
        empty_state("No risk signals", "None of the messages triggered rule-based signals.")
    else:
        fig = px.bar(x=sig.values, y=sig.index, orientation="h", title="Most frequent risk signals", color_discrete_sequence=[AMBER])
        fig.update_layout(height=360, xaxis_title="Messages", yaxis_title=None)
        st.plotly_chart(fig)
with g4:
    fig = px.scatter(res, x="words", y="spam_probability", color="verdict", color_discrete_map=VERDICT_COLORS,
                     hover_data={"message": False}, log_x=True, opacity=0.75, title="Message length vs spam probability")
    fig.update_layout(height=360, yaxis_tickformat=".0%", xaxis_title="Words (log scale)", legend_title=None)
    st.plotly_chart(fig)

if "actual" in res.columns:
    section("Accuracy on your labels", "Comparing the model's Spam / Not Spam decision with the label column you selected.")
    pred = (res.spam_probability >= t).map({True: "Spam", False: "Ham"})
    cm = confusion_matrix(res.actual, pred, labels=["Ham", "Spam"])
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / max(1, cm.sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    e1, e2 = st.columns([1.3, 1], gap="large")
    with e1:
        kpi_row([
            dict(label="Accuracy", value=f"{acc:.1%}", accent=BLUE),
            dict(label="Precision", value=f"{prec:.1%}", accent=AMBER),
            dict(label="Recall", value=f"{rec:.1%}", accent=GREEN),
            dict(label="F1", value=f"{2 * prec * rec / max(1e-9, prec + rec):.1%}", accent=RED),
        ])
        if "source" in res.columns:
            by = res.assign(correct=(pred == res.actual)).groupby("source").correct.mean().reset_index()
            fig = px.bar(by, x="source", y="correct", text_auto=".1%", title="Accuracy by source", color_discrete_sequence=[BLUE])
            fig.update_layout(height=280, yaxis=dict(tickformat=".0%", range=[0, 1.05]), xaxis_title=None, yaxis_title=None)
            st.plotly_chart(fig)
    with e2:
        fig = px.imshow(cm, x=["Pred Ham", "Pred Spam"], y=["Actual Ham", "Actual Spam"], text_auto=True,
                        color_continuous_scale=[[0, "#313244"], [1, BLUE]], title="Confusion matrix")
        fig.update_layout(height=360, coloraxis_showscale=False)
        st.plotly_chart(fig)

section("4. Results table")
f1c, f2c, f3c = st.columns([2, 2, 1])
pick = f1c.multiselect("Verdicts", ["Spam", "Suspicious", "Not Spam"], default=["Spam", "Suspicious", "Not Spam"])
query = f2c.text_input("Search messages", placeholder="e.g. prize, invoice, password")
min_p = f3c.number_input("Min probability", 0.0, 1.0, 0.0, 0.05)
view = res[res.verdict.isin(pick) & (res.spam_probability >= min_p)]
if query:
    view = view[view.message.str.contains(query, case=False, regex=False)]
view = view.sort_values("spam_probability", ascending=False)
st.caption(f"Showing {len(view):,} of {n:,} messages.")
st.dataframe(
    view.drop(columns=["phishing_pattern"]), hide_index=True, height=420,
    column_config={
        "message": st.column_config.TextColumn("Message", width="large"),
        "spam_probability": st.column_config.ProgressColumn("Spam probability", format="percent", min_value=0, max_value=1),
        "risk_score": st.column_config.ProgressColumn("Risk score", format="%d", min_value=0, max_value=100),
        "signals": st.column_config.TextColumn("Triggered signals", width="medium"),
    },
)
st.download_button("Download results (CSV)", res.drop(columns=["phishing_pattern"]).to_csv(index=False).encode("utf-8"),
                   file_name="spam_screening_results.csv", mime="text/csv", icon=":material/download:", type="primary")
