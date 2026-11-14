import streamlit as st

from ui.components import REPO_URL, get_detector, hero, info_card, section, steps_html
from ui.theme import BLUE, BORDER, CARD, CARD_2, GREEN, RED, TEXT

m = get_detector().metrics

hero(
    "How It Works",
    "An end-to-end machine-learning pipeline, from raw email to an explainable verdict, built as a "
    "GCET Jammu project by Ocean Gupta.",
    badges=["Python", "scikit-learn", "Streamlit", "Plotly", "NLP", "Explainable ML"],
)

section("Pipeline")
st.graphviz_chart(f"""
digraph {{
  rankdir=LR; bgcolor="transparent"; nodesep=0.35; ranksep=0.45;
  node [shape=box style="rounded,filled" fillcolor="{CARD}" color="{BORDER}" fontcolor="{TEXT}" fontname="Helvetica" fontsize=11 margin="0.18,0.1"];
  edge [color="{BLUE}" penwidth=1.6 arrowsize=0.7];
  raw [label="Raw message\\nSMS · email · HTML"];
  parse [label="Email parsing\\nheaders, HTML to text"];
  norm [label="Normalization\\nlinks, money, numbers,\\nCAPS, ! to tokens"];
  tfidf [label="TF-IDF\\n1-2 word n-grams\\n{m['vocabulary_size']:,} features"];
  svm [label="Linear SVM\\nclass-balanced"];
  cal [label="Calibration\\n5-fold sigmoid"];
  sig [label="Risk signals\\nphishing rules" fillcolor="{CARD_2}"];
  out [label="Verdict\\nSpam · Suspicious · Not Spam" fillcolor="#3b3150" color="{RED}"];
  raw -> parse -> norm -> tfidf -> svm -> cal -> out;
  parse -> sig -> out;
}}
""", width="stretch")

c1, c2, c3 = st.columns(3, gap="medium")
with c1:
    info_card("1 · Understand the text", steps_html([
        ("Parse", "Raw emails are split into sender, subject and body; HTML is stripped but link targets are kept."),
        ("Normalize", "Placeholders like [link], [money amount] and [phone] preserve signals that used to be deleted."),
    ]))
with c2:
    info_card("2 · Score it", steps_html([
        ("Vectorize", "TF-IDF weighs words and word pairs by how distinctive they are; sub-linear scaling tames repetition."),
        ("Classify", "A Linear SVM separates spam from ham; calibration turns its score into a real probability."),
    ]))
with c3:
    info_card("3 · Explain it", steps_html([
        ("Attribute", "Each feature's contribution (TF-IDF value × SVM weight) shows why a message was flagged."),
        ("Cross-check", "Transparent rules catch phishing that reads politely, e.g. 'update your payment information'."),
    ]))

section("The bug that was fixed", "Why the original app labelled almost everything 'Not Spam'.")
f1, f2 = st.columns(2, gap="large")
with f1:
    info_card("Root causes", """<ul>
      <li><b>Wrong data for the job:</b> trained only on 5.5k short SMS texts, while users paste emails, so email spam vocabulary was unknown.</li>
      <li><b>Signals thrown away:</b> preprocessing kept only alphanumeric tokens, deleting $, %, !, links and email addresses.</li>
      <li><b>Tiny vocabulary:</b> TF-IDF was capped at 3,000 single words, so most email terms were ignored.</li>
      <li><b>Imbalance:</b> 87% of SMS are ham and nothing compensated for it, so borderline messages defaulted to ham.</li>
      <li><b>Leaky evaluation:</b> the vectorizer was fitted before the train/test split and only SMS were tested, hiding the email failure.</li></ul>""")
with f2:
    e_old = m["baseline"]["by_source"]["Email"]["recall"]
    e_new = m["test_by_source"]["Email"]["recall"]
    info_card("Fixes & results", f"""<ul>
      <li>Added the SpamAssassin email corpus to the SMS data, deduplicated ({m['data']['total']:,} unique messages).</li>
      <li>Token-preserving normalization plus raw-email / HTML parsing.</li>
      <li>Class-balanced, calibrated Linear SVM on {m['vocabulary_size']:,} word + word-pair TF-IDF features, fitted on the training split only.</li>
      <li>Phishing / bulk-spam rule layer and an adjustable decision threshold.</li>
      <li>Versioned model artifacts, auto-retrained when stale.</li>
      <li><b style="color:{GREEN}">Email spam caught: {e_old:.0%} → {e_new:.0%}</b></li></ul>""")

section("Glossary")
g = st.columns(4, gap="medium")
terms = [
    ("Precision", "Of the messages flagged as spam, how many really were spam. High precision = few false alarms."),
    ("Recall", "Of all the real spam, how much was caught. High recall = little spam slips through."),
    ("F1 score", "The harmonic mean of precision and recall; a single balanced score."),
    ("ROC-AUC", "Probability that a random spam scores higher than a random ham. 1.0 is perfect, 0.5 is guessing."),
    ("TF-IDF", "Term frequency × inverse document frequency: rewards words that are frequent in a message but rare overall."),
    ("SVM", "Support Vector Machine: finds the boundary that separates the classes with the widest margin."),
    ("Calibration", "Adjusts raw classifier scores so that '80%' really means spam about 80% of the time."),
    ("Threshold", "The probability cut-off for a Spam verdict. Lower catches more spam but raises false alarms."),
]
for i, (t, d) in enumerate(terms):
    with g[i % 4]:
        info_card(t, f"<p>{d}</p>")
        st.write("")

section("Limitations")
info_card("Know before you trust", """<ul>
  <li>The email corpus (SpamAssassin, early 2000s) is dated; modern spam styles and non-English messages may be missed.</li>
  <li>Ham emails in SpamAssassin are largely technical mailing lists, so some list-specific words act as strong ham signals.</li>
  <li>Text-only: attachments, images, sender reputation and link destinations are not inspected.</li>
  <li>Risk signals are heuristics: they explain and flag, but can fire on legitimate security notices.</li></ul>""")

section("Project")
p1, p2, p3 = st.columns(3, gap="medium")
with p1:
    info_card("Author", "<p><b>Ocean Gupta</b><br>Government College of Engineering &amp; Technology (GCET), Jammu</p>")
with p2:
    info_card("Tech stack", "<p>Python · scikit-learn · NLTK · pandas · NumPy · Streamlit · Plotly · WordCloud · FastAPI (REST API)</p>")
with p3:
    info_card("Source & license", f'<p><a href="{REPO_URL}" target="_blank">{REPO_URL.replace("https://", "")}</a><br>MIT License</p>')
