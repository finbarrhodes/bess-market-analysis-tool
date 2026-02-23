"""
app.py — Streamlit Community Cloud entry point
===============================================
Run locally:
    streamlit run app.py
"""

import sys
from pathlib import Path

# Ensure src/ is importable when launched from any working directory
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

st.set_page_config(
    page_title="GB BESS Market Analysis",
    page_icon="⚡",
    layout="wide",
)

pages = [
    st.Page("src/visualization/dashboard.py",  title="Market Dashboard",    icon="⚡"),
    st.Page("src/visualization/backtester.py", title="Revenue Backtester",  icon="🔋"),
]

pg = st.navigation(pages)
pg.run()
