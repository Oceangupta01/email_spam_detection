import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

BG = "#1e1e2e"
CARD = "#2a2a3e"
CARD_2 = "#313244"
BORDER = "#45475a"
TEXT = "#cdd6f4"
MUTED = "#a6adc8"
BLUE = "#7c9ef5"
RED = "#f38ba8"
GREEN = "#a6e3a1"
AMBER = "#f9e2af"
MAUVE = "#cba6f7"
TEAL = "#94e2d5"
PEACH = "#fab387"

VERDICT_COLORS = {"Spam": RED, "Suspicious": AMBER, "Not Spam": GREEN}
CLASS_COLORS = {"Spam": RED, "Ham": GREEN}

pio.templates["spam_dark"] = go.layout.Template(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="Inter, Segoe UI, sans-serif", size=13),
        colorway=[BLUE, RED, GREEN, AMBER, MAUVE, TEAL, PEACH],
        xaxis=dict(gridcolor="rgba(166,173,200,0.12)", zerolinecolor="rgba(166,173,200,0.2)"),
        yaxis=dict(gridcolor="rgba(166,173,200,0.12)", zerolinecolor="rgba(166,173,200,0.2)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.0, x=0),
        title=dict(x=0, xanchor="left", y=0.985, yanchor="top", font=dict(size=15)),
        margin=dict(l=10, r=10, t=64, b=10),
        hoverlabel=dict(bgcolor=CARD_2, font=dict(color=TEXT)),
    )
)
pio.templates.default = "plotly_dark+spam_dark"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stText {{ font-family: 'Inter', 'Segoe UI', sans-serif; }}
.block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }}
h1, h2, h3, h4 {{ letter-spacing: -0.01em; }}

.hero {{
  background: radial-gradient(circle at 10% 0%, rgba(124,158,245,0.35), transparent 45%),
              radial-gradient(circle at 90% 100%, rgba(203,166,247,0.25), transparent 45%),
              linear-gradient(135deg, {CARD} 0%, {BG} 100%);
  border: 1px solid {BORDER}; border-radius: 20px; padding: 2rem 2.2rem; margin-bottom: 1.4rem;
}}
.hero h1 {{ font-size: 2.3rem; font-weight: 800; margin: 0 0 .4rem 0; color: {TEXT}; }}
.hero p {{ color: {MUTED}; font-size: 1.02rem; margin: 0; max-width: 820px; }}
.badges {{ margin-top: 1rem; display: flex; flex-wrap: wrap; gap: .5rem; }}
.badge {{ background: rgba(124,158,245,0.14); color: {BLUE}; border: 1px solid rgba(124,158,245,0.35);
  padding: .25rem .7rem; border-radius: 999px; font-size: .8rem; font-weight: 600; }}

.kpi {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 16px; padding: 1rem 1.2rem; height: 100%;
  position: relative; overflow: hidden; }}
.kpi::before {{ content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--accent); }}
.kpi .label {{ color: {MUTED}; font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; font-weight: 600; }}
.kpi .value {{ color: {TEXT}; font-size: 1.75rem; font-weight: 800; margin-top: .2rem; line-height: 1.15; }}
.kpi .sub {{ color: {MUTED}; font-size: .8rem; margin-top: .25rem; }}
.kpi .up {{ color: {GREEN}; font-weight: 700; }}
.kpi .down {{ color: {RED}; font-weight: 700; }}

.section-title {{ display: flex; align-items: center; gap: .6rem; margin: 1.6rem 0 .2rem 0; }}
.section-title h3 {{ margin: 0; font-size: 1.25rem; font-weight: 700; }}
.section-sub {{ color: {MUTED}; font-size: .9rem; margin-bottom: .8rem; }}

.verdict {{ border-radius: 18px; padding: 1.4rem 1.6rem; border: 1px solid var(--accent);
  background: linear-gradient(135deg, color-mix(in srgb, var(--accent) 22%, {CARD}) 0%, {CARD} 70%); }}
.verdict .tag {{ color: var(--accent); font-weight: 700; letter-spacing: .08em; font-size: .8rem; text-transform: uppercase; }}
.verdict .title {{ font-size: 2rem; font-weight: 800; color: {TEXT}; margin: .2rem 0; }}
.verdict .desc {{ color: {MUTED}; font-size: .95rem; }}

.signal {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 14px; padding: .8rem 1rem; margin-bottom: .7rem; }}
.signal.on {{ border-color: rgba(243,139,168,0.6); background: linear-gradient(135deg, rgba(243,139,168,0.10), {CARD}); }}
.signal .head {{ display: flex; justify-content: space-between; align-items: center; font-weight: 600; color: {TEXT}; }}
.signal .pts {{ font-size: .75rem; padding: .1rem .5rem; border-radius: 999px; background: {CARD_2}; color: {MUTED}; }}
.signal.on .pts {{ background: rgba(243,139,168,0.2); color: {RED}; }}
.signal .desc {{ color: {MUTED}; font-size: .82rem; margin-top: .25rem; }}
.chip {{ display: inline-block; background: {CARD_2}; color: {TEXT}; border-radius: 6px; padding: .05rem .45rem;
  margin: .3rem .3rem 0 0; font-size: .75rem; font-family: ui-monospace, monospace; }}

.highlight-box {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 14px; padding: 1rem 1.2rem;
  line-height: 1.9; max-height: 360px; overflow-y: auto; white-space: pre-wrap; word-break: break-word; }}
.highlight-box span {{ border-radius: 4px; padding: 0 2px; }}
.legend {{ color: {MUTED}; font-size: .8rem; margin-top: .4rem; }}

.info-card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 16px; padding: 1.1rem 1.3rem; height: 100%; }}
.info-card h4 {{ margin: 0 0 .4rem 0; font-size: 1rem; color: {TEXT}; }}
.info-card p, .info-card li {{ color: {MUTED}; font-size: .9rem; }}
.step {{ display: flex; gap: .8rem; align-items: flex-start; margin-bottom: .8rem; }}
.step .n {{ min-width: 28px; height: 28px; border-radius: 50%; background: rgba(124,158,245,0.18); color: {BLUE};
  display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: .85rem; }}
.step .t {{ color: {TEXT}; font-weight: 600; font-size: .92rem; }}
.step .d {{ color: {MUTED}; font-size: .82rem; }}

.app-footer {{ text-align: center; color: {MUTED}; font-size: .8rem; margin-top: 3rem; padding-top: 1rem;
  border-top: 1px solid {BORDER}; }}
.app-footer a {{ color: {BLUE}; text-decoration: none; }}
div[data-testid="stTabs"] button p {{ font-weight: 600; }}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)
