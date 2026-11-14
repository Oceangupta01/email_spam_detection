import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ui.components import get_detector, hero, info_card, kpi_row, section, threshold
from ui.theme import AMBER, BLUE, CLASS_COLORS, GREEN, MAUVE, MUTED, RED, TEAL

m = get_detector().metrics
test, base = m["test"], m["baseline"]

hero(
    "Model Performance",
    "Every number on this page is measured on a held-out test set of messages the model never saw during training, "
    "after removing duplicate messages so results aren't inflated.",
    badges=[f"{m['data']['test']:,} test messages", f"Trained {m['trained_at'][:10]}", f"Model v{m['model_version']}",
            f"Training time {m['training_seconds']}s", f"scikit-learn {m['sklearn_version']}"],
)

kpi_row([
    dict(label="Accuracy", value=f"{test['accuracy']:.2%}", sub="vs original", delta=test["accuracy"] - base["overall"]["accuracy"], accent=BLUE),
    dict(label="Precision", value=f"{test['precision']:.2%}", sub="vs original", delta=test["precision"] - base["overall"]["precision"], accent=AMBER),
    dict(label="Recall (spam caught)", value=f"{test['recall']:.2%}", sub="vs original", delta=test["recall"] - base["overall"]["recall"], accent=GREEN),
    dict(label="F1 score", value=f"{test['f1']:.2%}", sub="vs original", delta=test["f1"] - base["overall"]["f1"], accent=RED),
    dict(label="ROC-AUC", value=f"{test['roc_auc']:.4f}", sub="vs original", delta=test["roc_auc"] - base["overall"]["roc_auc"], delta_fmt="{:+.3f}", accent=MAUVE),
])

section("Before vs after the fix",
        "The original app trained only on SMS texts and deleted digits, symbols and links, so it missed about half of real email spam.")
rows = []
for src in ("SMS", "Email"):
    for name, block in (("Original model", base["by_source"][src]), ("Improved model", m["test_by_source"][src])):
        for metric in ("accuracy", "precision", "recall", "f1"):
            rows.append({"source": src, "model": name, "metric": metric.title() if metric != "f1" else "F1", "value": block[metric]})
bdf = pd.DataFrame(rows)
b1, b2 = st.columns([2, 1], gap="large")
with b1:
    fig = px.bar(bdf, x="metric", y="value", color="model", barmode="group", facet_col="source", text_auto=".1%",
                 color_discrete_map={"Original model": "#6c7086", "Improved model": BLUE})
    fig.for_each_annotation(lambda a: a.update(text=a.text.replace("source=", "") + " messages"))
    fig.update_layout(height=400, yaxis=dict(tickformat=".0%", range=[0, 1.1]), yaxis2=dict(tickformat=".0%"), legend_title=None,
                      legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center", yanchor="top"))
    fig.update_xaxes(title=None)
    fig.update_yaxes(title=None)
    st.plotly_chart(fig)
with b2:
    e_old, e_new = base["by_source"]["Email"], m["test_by_source"]["Email"]
    missed_old = e_old["confusion_matrix"][1][0]
    missed_new = e_new["confusion_matrix"][1][0]
    info_card("What changed", f"""
      <ul>
        <li><b>Training data:</b> SMS + SpamAssassin emails ({m['data']['total']:,} unique messages) instead of SMS only.</li>
        <li><b>Preprocessing:</b> links, money, phone numbers, %, ALL-CAPS and "!" kept as tokens instead of deleted.</li>
        <li><b>Features:</b> word pairs (bigrams) and sub-linear TF-IDF.</li>
        <li><b>Class balance:</b> spam weighted to offset the 3:1 ham majority.</li>
        <li><b>Result:</b> missed email spam dropped from <b>{missed_old}</b> to <b>{missed_new}</b> of {e_new['support']['spam']} in the test set.</li>
      </ul>""")

section("Error analysis")
c1, c2 = st.columns([1, 1.4], gap="large")
with c1:
    scope = st.radio("Confusion matrix for", ["All", "SMS", "Email"], horizontal=True)
    block = test if scope == "All" else m["test_by_source"][scope]
    cm = block["confusion_matrix"]
    labels = [[f"{v:,}<br>{v / max(1, sum(r)):.1%} of row" for v in r] for r in cm]
    fig = go.Figure(go.Heatmap(z=cm, x=["Predicted Ham", "Predicted Spam"], y=["Actual Ham", "Actual Spam"], text=labels,
                               texttemplate="%{text}", colorscale=[[0, "#313244"], [1, BLUE]], showscale=False))
    fig.update_layout(height=360, yaxis_autorange="reversed")
    st.plotly_chart(fig)
with c2:
    probs = pd.DataFrame(m["test_probabilities"])
    probs["class"] = probs.label.map({0: "Ham", 1: "Spam"})
    fig = px.histogram(probs, x="prob", color="class", nbins=50, barmode="overlay", opacity=0.75, log_y=True,
                       color_discrete_map=CLASS_COLORS, title="How confident is the model? (test-set probabilities)")
    fig.add_vline(x=threshold(), line_dash="dash", line_color=MUTED, annotation_text=f"threshold {threshold():.0%}")
    fig.update_layout(height=400, xaxis_title="Predicted spam probability", yaxis_title="Messages (log)", xaxis_tickformat=".0%", legend_title=None)
    st.plotly_chart(fig)

section("Curves", "ROC shows the trade-off between catching spam and false alarms; precision-recall focuses on the spam class.")
r1, r2, r3 = st.columns(3, gap="medium")
with r1:
    fig = go.Figure()
    fig.add_scatter(x=m["roc_curve"]["fpr"], y=m["roc_curve"]["tpr"], mode="lines", fill="tozeroy", line=dict(color=BLUE, width=3),
                    name=f"AUC {test['roc_auc']:.4f}", fillcolor="rgba(124,158,245,0.15)")
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=MUTED, dash="dash"), name="Random")
    fig.update_layout(height=360, title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
    st.plotly_chart(fig)
with r2:
    fig = go.Figure()
    fig.add_scatter(x=m["pr_curve"]["recall"], y=m["pr_curve"]["precision"], mode="lines", fill="tozeroy",
                    line=dict(color=GREEN, width=3), name=f"AP {test['average_precision']:.4f}", fillcolor="rgba(166,227,161,0.15)")
    fig.update_layout(height=360, title="Precision-recall curve", xaxis_title="Recall", yaxis_title="Precision", yaxis_range=[0, 1.02])
    st.plotly_chart(fig)
with r3:
    sw = pd.DataFrame(m["threshold_sweep"]).melt("threshold", var_name="metric", value_name="value")
    fig = px.line(sw, x="threshold", y="value", color="metric", markers=True, title="Threshold tuning",
                  color_discrete_map={"precision": AMBER, "recall": GREEN, "f1": BLUE})
    fig.add_vline(x=threshold(), line_dash="dash", line_color=MUTED)
    fig.update_layout(height=360, yaxis=dict(tickformat=".0%"), xaxis_tickformat=".0%", legend_title=None)
    st.plotly_chart(fig)

section("Algorithm comparison", "Five classifiers trained on the same features (the notebook's model comparison, re-run on the new data).")
cmp = pd.DataFrame(m["model_comparison"])
k1, k2 = st.columns([1.5, 1], gap="large")
with k1:
    long = cmp.melt("model", ["accuracy", "precision", "recall", "f1"], var_name="metric", value_name="value")
    long["metric"] = long.metric.str.replace("f1", "F1").str.title().str.replace("F1", "F1")
    fig = px.bar(long, x="value", y="model", color="metric", barmode="group", orientation="h",
                 color_discrete_sequence=[BLUE, AMBER, GREEN, RED])
    fig.update_layout(height=430, xaxis=dict(tickformat=".0%", range=[0.8, 1.0]), yaxis_title=None, xaxis_title=None, legend_title=None)
    st.plotly_chart(fig)
with k2:
    st.dataframe(
        cmp[["model", "accuracy", "precision", "recall", "f1", "roc_auc", "train_seconds", "predict_ms_per_msg"]],
        hide_index=True, height=230,
        column_config={
            "model": "Model",
            "accuracy": st.column_config.NumberColumn("Accuracy", format="percent"),
            "precision": st.column_config.NumberColumn("Precision", format="percent"),
            "recall": st.column_config.NumberColumn("Recall", format="percent"),
            "f1": st.column_config.NumberColumn("F1", format="percent"),
            "roc_auc": st.column_config.NumberColumn("ROC-AUC", format="%.4f"),
            "train_seconds": st.column_config.NumberColumn("Train (s)", format="%.2f"),
            "predict_ms_per_msg": st.column_config.NumberColumn("ms / msg", format="%.3f"),
        },
    )
    st.caption("The deployed model is the calibrated Linear SVM: it has the best ROC-AUC with balanced precision and recall, "
               "produces reliable probabilities, and is fast and explainable through its word weights.")

section("What the model learned", "Features with the largest weights; bracketed items are placeholders for links, numbers, etc.")
spam_t = pd.DataFrame(m["top_spam_terms"][:20])
ham_t = pd.DataFrame(m["top_ham_terms"][:20])
w1, w2 = st.columns(2, gap="large")
with w1:
    fig = px.bar(spam_t.iloc[::-1], x="weight", y="term", orientation="h", title="Strongest spam indicators", color_discrete_sequence=[RED])
    fig.update_layout(height=560, yaxis_title=None, xaxis_title="SVM weight")
    st.plotly_chart(fig)
with w2:
    fig = px.bar(ham_t.iloc[::-1].assign(weight=lambda d: d.weight.abs()), x="weight", y="term", orientation="h",
                 title="Strongest ham (legitimate) indicators", color_discrete_sequence=[GREEN])
    fig.update_layout(height=560, yaxis_title=None, xaxis_title="SVM weight (absolute)")
    st.plotly_chart(fig)

section("Hardest test messages", "Misclassified test messages, closest-to-confident mistakes first.")
st.dataframe(
    pd.DataFrame(m["misclassified"]), hide_index=True, height=360,
    column_config={
        "text": st.column_config.TextColumn("Message", width="large"),
        "spam_probability": st.column_config.ProgressColumn("Spam probability", format="percent", min_value=0, max_value=1),
    },
)

d = m["data"]["by_source"]
section("Dataset split")
s1, s2, s3 = st.columns(3)
with s1:
    info_card("Training / test", f"<p><b>{m['data']['train']:,}</b> training and <b>{m['data']['test']:,}</b> test messages "
              f"(80 / 20 split, stratified by source and class).</p>")
with s2:
    info_card("SMS Spam Collection", f"<p><b>{d['SMS']['ham']:,}</b> ham · <b>{d['SMS']['spam']:,}</b> spam (unique)</p>")
with s3:
    info_card("SpamAssassin emails", f"<p><b>{d['Email']['ham']:,}</b> ham · <b>{d['Email']['spam']:,}</b> spam (unique)</p>")
