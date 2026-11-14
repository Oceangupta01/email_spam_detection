import re

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer

from data_preprocessing import CAPS, MONEY, PERCENT, PHONE, URL, pretty_term
from ui.components import get_datasets, hero, kpi_row, section
from ui.theme import AMBER, BLUE, CLASS_COLORS, GREEN, MAUVE, RED

try:
    from wordcloud import WordCloud
except ImportError:
    WordCloud = None

df = get_datasets()

FEATURES = {
    "Contains a link": URL,
    "Mentions money": MONEY,
    "Phone number": PHONE,
    "Percentage": PERCENT,
    "ALL-CAPS word": CAPS,
    "Exclamation mark": re.compile("!"),
    "Word 'free'": re.compile(r"\bfree\b", re.I),
    "Word 'click'": re.compile(r"\bclick\b", re.I),
}


@st.cache_data(show_spinner=False)
def feature_presence(source: str) -> pd.DataFrame:
    d = df if source == "All" else df[df.source == source]
    rows = []
    for name, pattern in FEATURES.items():
        hit = d.text.map(lambda t, p=pattern: p.search(t) is not None)
        for cls in ("Spam", "Ham"):
            rows.append({"feature": name, "class": cls, "share": hit[d.label_name == cls].mean()})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Counting terms...")
def top_terms(source: str, ngram: int, n: int = 20) -> pd.DataFrame:
    d = df if source == "All" else df[df.source == source]
    vec = CountVectorizer(token_pattern=r"\S+", ngram_range=(ngram, ngram), min_df=3, binary=True)
    X = vec.fit_transform(d.clean)
    names = vec.get_feature_names_out()
    rows = []
    for cls, lab in (("Spam", 1), ("Ham", 0)):
        mask = (d.label == lab).values
        share = pd.Series(X[mask].mean(axis=0).A1, index=names).nlargest(n)
        rows += [{"class": cls, "term": pretty_term(t), "share": s} for t, s in share.items()]
    return pd.DataFrame(rows)


hero(
    "Dataset Insights",
    "Explore the two public corpora behind the model, the SMS Spam Collection and the SpamAssassin email corpus: "
    "class balance, message lengths, the vocabulary of spam, and the tell-tale features spammers use.",
    badges=["SMS Spam Collection (UCI)", "SpamAssassin public corpus", f"{len(df):,} unique messages", "Duplicates removed"],
)

spam_share = df.label.mean()
med = df.groupby("label_name").words.median()
kpi_row([
    dict(label="Unique messages", value=f"{len(df):,}", sub=f"{(df.source == 'SMS').sum():,} SMS · {(df.source == 'Email').sum():,} email", accent=BLUE),
    dict(label="Spam share", value=f"{spam_share:.1%}", sub=f"{df.label.sum():,} spam messages", accent=RED),
    dict(label="Median spam length", value=f"{med['Spam']:.0f} words", sub="all sources", accent=AMBER),
    dict(label="Median ham length", value=f"{med['Ham']:.0f} words", sub="all sources", accent=GREEN),
])

section("Composition")
c1, c2 = st.columns(2, gap="large")
with c1:
    comp = df.groupby(["source", "label_name"]).size().reset_index(name="messages")
    fig = px.sunburst(comp, path=["source", "label_name"], values="messages", color="label_name",
                      color_discrete_map={**CLASS_COLORS, "(?)": MAUVE}, title="Messages by source and class")
    fig.update_traces(insidetextorientation="horizontal", textinfo="label+value+percent parent")
    fig.update_layout(height=400)
    st.plotly_chart(fig)
with c2:
    fig = px.histogram(comp, x="source", y="messages", color="label_name", barnorm="percent", text_auto=".1f",
                       color_discrete_map=CLASS_COLORS, title="Class balance per source (%)")
    fig.update_layout(height=400, yaxis_title="% of messages", xaxis_title=None, legend_title=None)
    st.plotly_chart(fig)

section("Message length", "Spam SMS tend to be long (they use all 160 characters), while spam emails are often shorter than mailing-list ham.")
l1, l2 = st.columns(2, gap="large")
with l1:
    fig = px.box(df, x="source", y="words", color="label_name", log_y=True, color_discrete_map=CLASS_COLORS, points=False,
                 title="Words per message (log scale)")
    fig.update_layout(height=400, xaxis_title=None, legend_title=None)
    st.plotly_chart(fig)
with l2:
    clip = df[df.characters <= df.characters.quantile(0.95)]
    fig = px.histogram(clip, x="characters", color="label_name", facet_row="source", nbins=60, barmode="overlay", opacity=0.75,
                       color_discrete_map=CLASS_COLORS, title="Character count distribution (95th percentile cut)")
    fig.for_each_annotation(lambda a: a.update(text=a.text.replace("source=", ""), textangle=0, x=0.98, xanchor="right"))
    fig.update_yaxes(matches=None, title=None)
    fig.update_layout(height=400, legend_title=None, xaxis_title="characters")
    st.plotly_chart(fig)

section("Tell-tale features", "Share of messages in each class that contain a given feature.")
src = st.segmented_control("Source", ["All", "SMS", "Email"], default="All", key="insight_source") or "All"
fp = feature_presence(src)
fig = px.bar(fp, x="share", y="feature", color="class", barmode="group", orientation="h", text_auto=".0%",
             color_discrete_map=CLASS_COLORS)
fig.update_layout(height=440, xaxis=dict(tickformat=".0%"), yaxis_title=None, xaxis_title="Share of messages", legend_title=None)
st.plotly_chart(fig)

section("Vocabulary", "Most common stemmed words / word pairs per class, as share of messages containing them (stop words removed, links, money, phone numbers etc. shown as [placeholders]).")
ngram_label = st.segmented_control("Terms", ["Single words", "Word pairs"], default="Single words", key="insight_ngram") or "Single words"
tt = top_terms(src, 1 if ngram_label == "Single words" else 2)
v1, v2 = st.columns(2, gap="large")
for col, cls, color in ((v1, "Spam", RED), (v2, "Ham", GREEN)):
    with col:
        part = tt[tt["class"] == cls].iloc[::-1]
        fig = px.bar(part, x="share", y="term", orientation="h", title=f"Top {cls.lower()} terms", color_discrete_sequence=[color])
        fig.update_layout(height=560, xaxis=dict(tickformat=".0%"), yaxis_title=None, xaxis_title="Share of messages")
        st.plotly_chart(fig)

if WordCloud is not None:
    section("Word clouds")
    w1, w2 = st.columns(2, gap="large")
    big = top_terms(src, 1, n=120)
    for col, cls, cmap in ((w1, "Spam", "Reds"), (w2, "Ham", "Greens")):
        part = big[big["class"] == cls]
        freqs = dict(zip(part.term, part.share))
        img = WordCloud(width=900, height=420, background_color=None, mode="RGBA", colormap=cmap,
                        prefer_horizontal=0.9, random_state=3).generate_from_frequencies(freqs)
        with col:
            st.markdown(f"**{cls}**")
            st.image(img.to_array(), width="stretch")

section("Message explorer", "Browse the raw training data.")
e1, e2, e3 = st.columns([1, 1, 2])
f_src = e1.multiselect("Source", ["SMS", "Email"], default=["SMS", "Email"])
f_cls = e2.multiselect("Class", ["Spam", "Ham"], default=["Spam"])
f_q = e3.text_input("Contains", placeholder="e.g. lottery, meeting, unsubscribe")
view = df[df.source.isin(f_src) & df.label_name.isin(f_cls)]
if f_q:
    view = view[view.text.str.contains(f_q, case=False, regex=False)]
st.caption(f"{len(view):,} matching messages · showing up to 300")
st.dataframe(
    view[["text", "label_name", "source", "split", "words"]].head(300), hide_index=True, height=380,
    column_config={"text": st.column_config.TextColumn("Message", width="large"), "label_name": "Class",
                   "source": "Source", "split": "Split", "words": "Words"},
)
