"""Streamlit entrypoint.

Run:  streamlit run streamlit_app.py
Needs NEBIUS_API_KEY in .env, models/ from cv/fit_patchcore.ipynb, and data/mvtec (python -m cv.download_mvtec --foreign).
"""
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

st.set_page_config(page_title="Warranty Claims Agent", page_icon=":material/fact_check:", layout="wide")

st.navigation([
    st.Page("app_pages/claims.py", title="Warranty claims", icon=":material/fact_check:", default=True),
    st.Page("app_pages/evaluation.py", title="Evaluation", icon=":material/monitoring:"),
    st.Page("app_pages/gallery.py", title="Heatmap gallery", icon=":material/grid_view:"),
    st.Page("app_pages/claims_analyst.py", title="Claims analyst", icon=":material/query_stats:"),
]).run()
