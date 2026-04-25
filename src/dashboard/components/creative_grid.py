"""Creative grid component with performance indicators."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from loguru import logger

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

_GOOD_STATUSES = {"top_performer", "stable"}
_BAD_STATUSES = {"underperformer", "fatigued"}


def _asset_path(asset_file: str) -> Path:
    return _PROJECT_ROOT / "data" / asset_file


def _smiley(status: str) -> str:
    if status in _GOOD_STATUSES:
        return "😊"
    return "😔"


def _status_color(status: str) -> str:
    if status in _GOOD_STATUSES:
        return "green"
    return "red"


def render_creative_grid(campaign_summary: pd.DataFrame) -> None:
    """Render a grid of creatives with smiley/sad performance indicators.

    Parameters
    ----------
    campaign_summary:
        creative_summary rows for the selected campaign (already ID-mapped).
    """
    if campaign_summary.empty:
        st.info("No creatives found for this campaign.")
        return

    sorted_df = campaign_summary.sort_values("creative_id").reset_index(drop=True)
    n_cols = 6
    rows = [sorted_df.iloc[i : i + n_cols] for i in range(0, len(sorted_df), n_cols)]

    st.subheader("Creatives")

    for row_df in rows:
        cols = st.columns(n_cols)
        for col, (_, creative_row) in zip(cols, row_df.iterrows()):
            creative_id = creative_row["creative_id"]
            status = str(creative_row.get("creative_status", "stable"))
            perf = float(creative_row.get("perf_score", 0.0))
            asset_file = creative_row.get("asset_file", "")
            img_path = _asset_path(asset_file) if asset_file else Path("/nonexistent")

            smiley = _smiley(status)
            color = _status_color(status)

            with col:
                with st.container(border=True):
                    if img_path.exists():
                        st.image(str(img_path), use_container_width=True)
                    else:
                        st.markdown(
                            "<div style='height:120px;background:#eee;display:flex;"
                            "align-items:center;justify-content:center;border-radius:4px'>"
                            "No image</div>",
                            unsafe_allow_html=True,
                        )

                    bottom_left, bottom_right = st.columns([1, 2])
                    with bottom_left:
                        st.markdown(
                            f"<span style='font-size:1.4rem'>{smiley}</span>",
                            unsafe_allow_html=True,
                        )
                    with bottom_right:
                        st.markdown(
                            f"<span style='color:{color};font-size:0.85rem'>"
                            f"**{creative_id}**<br>{perf:.3f}</span>",
                            unsafe_allow_html=True,
                        )

                    if st.button(
                        "View Ad", key=f"creative_btn_{creative_id}", use_container_width=True
                    ):
                        st.session_state.current_view = "ad_detail"
                        st.session_state.selected_creative = creative_id
                        logger.debug("Navigating to ad detail: {}", creative_id)
                        st.rerun()
