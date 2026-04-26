"""Global CSS injection for the Smadex dashboard."""

from __future__ import annotations

import streamlit as st

_PURPLE = "#881CA6"
_PURPLE_BORDER = "#c97de0"

_CSS = f"""
<style>
/* ── Slightly darker background (works in light mode) ───────────────── */
[data-testid="stAppViewContainer"] > .main {{
    background: #ede8f2;
}}
[data-testid="stHeader"] {{
    background: transparent;
}}

/* ── Keep container borders but no white override ───────────────────── */
[data-testid="stVerticalBlockBorderWrapper"] {{
    border: 1px solid {_PURPLE_BORDER}55 !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 6px rgba(136,28,166,0.06) !important;
}}

/* ── Plotly chart slight border ─────────────────────────────────────── */
[data-testid="stPlotlyChart"] {{
    border-radius: 10px;
}}

/* ── Metric label styling ───────────────────────────────────────────── */
[data-testid="stMetricLabel"] > div {{
    font-size: 12px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    opacity: 0.7;
}}

/* ── Selectbox border ───────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {{
    border-color: {_PURPLE_BORDER} !important;
    border-radius: 8px !important;
}}

/* ── Divider ────────────────────────────────────────────────────────── */
hr {{
    border-color: {_PURPLE_BORDER}66 !important;
}}
</style>
"""


def inject_css() -> None:
    """Call once at the top of the main render function."""
    st.markdown(_CSS, unsafe_allow_html=True)
