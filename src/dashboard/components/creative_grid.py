"""Creative grid component with performance indicators."""

from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd
import streamlit as st
from loguru import logger
from PIL import Image as PILImage

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

_GOOD_STATUSES = {"top_performer", "stable"}

_STATUS_COLORS = {
    "top_performer": "#16a34a",
    "stable": "#2563eb",
    "underperformer": "#d97706",
    "fatigued": "#dc2626",
}
_STATUS_BG = {
    "top_performer": "#f0fdf4",
    "stable": "#eff6ff",
    "underperformer": "#fffbeb",
    "fatigued": "#fef2f2",
}


def _asset_path(asset_file: str) -> Path:
    return _PROJECT_ROOT / "data" / asset_file


def _max_display_height(img_paths: list[Path], ref_width: int = 200) -> int:
    """Return the tallest display height when all images are scaled to ref_width."""
    max_h = 80
    for p in img_paths:
        if not p.exists():
            continue
        try:
            with PILImage.open(p) as img:
                w, h = img.size
                if w > 0:
                    max_h = max(max_h, int(h * ref_width / w))
        except Exception:
            pass
    return min(max_h, 400)


def _img_html(img_path: Path, height: int) -> str:
    data = base64.b64encode(img_path.read_bytes()).decode()
    ext = img_path.suffix.lstrip(".")
    return (
        f'<div style="width:100%;height:{height}px;overflow:hidden;border-radius:4px;'
        f'background:#f5e9f9;display:flex;align-items:center;justify-content:center">'
        f'<img src="data:image/{ext};base64,{data}" '
        f'style="width:100%;height:100%;object-fit:contain"/>'
        f"</div>"
    )


def render_creative_grid(campaign_summary: pd.DataFrame) -> None:
    if campaign_summary.empty:
        st.info("No creatives found for this campaign.")
        return

    sorted_df = campaign_summary.sort_values("creative_id").reset_index(drop=True)
    n_cols = 6
    rows = [sorted_df.iloc[i : i + n_cols] for i in range(0, len(sorted_df), n_cols)]

    img_paths = [
        _asset_path(r.get("asset_file", ""))
        for _, r in sorted_df.iterrows()
        if r.get("asset_file")
    ]
    img_height = _max_display_height(img_paths)

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
            bg = _STATUS_BG.get(status, "#f9fafb")
            label = status.replace("_", " ").title()

            with col:
                with st.container(border=True):
                    # Status colour strip at top
                    st.markdown(
                        f'<div style="height:4px;background:{color};'
                        f'border-radius:4px;margin-bottom:6px"></div>',
                        unsafe_allow_html=True,
                    )
                    if img_path.exists():
                        st.markdown(_img_html(img_path, img_height), unsafe_allow_html=True)
                    else:
                        st.markdown(
                            f'<div style="height:{img_height}px;background:#f5e9f9;display:flex;'
                            "align-items:center;justify-content:center;border-radius:4px;"
                            'color:#881CA6;font-size:11px">No image</div>',
                            unsafe_allow_html=True,
                        )

                    st.markdown(
                        f'<div style="display:flex;align-items:center;'
                        f'justify-content:space-between;margin-top:6px;flex-wrap:wrap;gap:2px">'
                        f'<div style="display:inline-flex;align-items:center;'
                        f"background:{bg};border:1px solid {color}40;color:{color};"
                        f"padding:2px 7px;border-radius:20px;font-weight:600;"
                        f'font-size:10px;white-space:nowrap">{label}</div>'
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
