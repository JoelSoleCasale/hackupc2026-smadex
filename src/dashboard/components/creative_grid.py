"""Creative grid component with performance indicators."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from loguru import logger

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

_GOOD_STATUSES = {"top_performer", "stable"}

_STATUS_COLORS = {
    "top_performer": "#16a34a",
    "stable": "#2563eb",
    "underperformer": "#d97706",
    "fatigued": "#dc2626",
}
_STATUS_EMOJI = {
    "top_performer": "😊",
    "stable": "😊",
    "underperformer": "😔",
    "fatigued": "😔",
}


def _asset_path(asset_file: str) -> Path:
    return _PROJECT_ROOT / "data" / asset_file


def render_creative_grid(campaign_summary: pd.DataFrame) -> None:
    if campaign_summary.empty:
        st.info("No creatives found for this campaign.")
        return

    sorted_df = campaign_summary.sort_values("creative_id").reset_index(drop=True)
    n_cols = 6
    rows = [sorted_df.iloc[i : i + n_cols] for i in range(0, len(sorted_df), n_cols)]

    st.markdown(
        '<div style="border-left:4px solid #881CA6;padding-left:10px;margin:16px 0 10px">'
        '<span style="font-size:17px;font-weight:700">Creatives</span></div>',
        unsafe_allow_html=True,
    )

    for row_df in rows:
        cols = st.columns(n_cols)
        for col, (_, creative_row) in zip(cols, row_df.iterrows()):
            creative_id = creative_row["creative_id"]
            status = str(creative_row.get("creative_status", "stable"))
            perf = float(creative_row.get("perf_score", 0.0))
            asset_file = creative_row.get("asset_file", "")
            img_path = _asset_path(asset_file) if asset_file else Path("/nonexistent")

            color = _STATUS_COLORS.get(status, "#6b7280")
            emoji = _STATUS_EMOJI.get(status, "😐")

            with col:
                with st.container(border=True):
                    # Status colour strip at top
                    st.markdown(
                        f'<div style="height:4px;background:{color};'
                        f'border-radius:4px;margin-bottom:6px"></div>',
                        unsafe_allow_html=True,
                    )
                    if img_path.exists():
                        st.image(str(img_path), use_container_width=True)
                    else:
                        st.markdown(
                            '<div style="height:100px;background:#f5e9f9;display:flex;'
                            "align-items:center;justify-content:center;border-radius:4px;"
                            'color:#881CA6;font-size:11px">No image</div>',
                            unsafe_allow_html=True,
                        )

                    st.markdown(
                        f'<div style="display:flex;align-items:center;justify-content:space-between;'
                        f'margin-top:4px">'
                        f'<span style="font-size:1.2rem">{emoji}</span>'
                        f'<span style="color:{color};font-size:11px;font-weight:700">'
                        f"{perf:.3f}</span></div>",
                        unsafe_allow_html=True,
                    )

                    if st.button(
                        "View Ad", key=f"creative_btn_{creative_id}", use_container_width=True
                    ):
                        st.session_state.current_view = "ad_detail"
                        st.session_state.selected_creative = creative_id
                        logger.debug("Navigating to ad detail: {}", creative_id)
                        st.rerun()
