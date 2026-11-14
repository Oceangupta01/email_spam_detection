"""
Email / SMS Spam Detection System – Streamlit web app.

Run locally:  streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui.components import footer, get_detector, sidebar  # noqa: E402
from ui.theme import inject_css  # noqa: E402

inject_css()
sidebar(get_detector())

page = st.navigation(
    {
        "Analyze": [
            st.Page("app_pages/detector.py", title="Spam Detector", icon=":material/shield:", default=True),
            st.Page("app_pages/batch.py", title="Batch Analysis", icon=":material/table_view:"),
        ],
        "Insights": [
            st.Page("app_pages/performance.py", title="Model Performance", icon=":material/monitoring:"),
            st.Page("app_pages/insights.py", title="Dataset Insights", icon=":material/query_stats:"),
        ],
        "Project": [
            st.Page("app_pages/about.py", title="How It Works", icon=":material/info:"),
        ],
    }
)
page.run()
footer()
