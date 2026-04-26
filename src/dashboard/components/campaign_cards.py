"""Campaign cards component for the Smadex Creative Intelligence dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from loguru import logger

_PURPLE = "#881CA6"
_PURPLE_LIGHT = "#f5e9f9"


def _metric_box(label: str, value: str, *, purple: bool = False) -> str:
    if purple:
        bg, border, label_color, val_color = _PURPLE_LIGHT, "#d4a8e6", _PURPLE, _PURPLE
    else:
        bg, border, label_color, val_color = "#f3f4f6", "#d1d5db", "#6b7280", "#374151"
    return (
        f'<div style="background:{bg};border:1px solid {border};border-radius:10px;'
        f'padding:8px 12px;text-align:center;margin-bottom:6px">'
        f'<div style="font-size:10px;color:{label_color};font-weight:600;margin-bottom:2px">{label}</div>'
        f'<div style="font-size:17px;font-weight:700;color:{val_color}">{value}</div>'
        f"</div>"
    )


def render_campaign_cards(adv_summary: pd.DataFrame, adv_campaigns: pd.DataFrame) -> None:
    campaigns = sorted(adv_summary["campaign_id"].unique())
    logger.debug("render_campaign_cards: {} campaigns", len(campaigns))

    if not campaigns:
        st.info("No campaigns found for this advertiser.")
        return

    st.markdown(
        f'<div style="border-left:4px solid {_PURPLE};padding-left:10px;margin:16px 0 10px">'
        f'<span style="font-size:17px;font-weight:700">Campaigns</span></div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(len(campaigns))

    for col, camp_id in zip(cols, campaigns):
        camp_summary = adv_summary[adv_summary["campaign_id"] == camp_id]
        avg_perf = float(camp_summary["perf_score"].mean()) if not camp_summary.empty else 0.0
        camp_meta = adv_campaigns[adv_campaigns["campaign_id"] == camp_id]
        objective = camp_meta["objective"].iloc[0] if not camp_meta.empty else "—"
        kpi_goal = camp_meta["kpi_goal"].iloc[0] if not camp_meta.empty else "—"
        n_creatives = len(camp_summary)

        with col:
            with st.container(border=True):
                st.markdown(
                    f'<div style="background:linear-gradient(90deg,{_PURPLE},{_PURPLE}bb);'
                    f'padding:8px 12px;border-radius:6px;margin-bottom:10px">'
                    f'<span style="color:#fff;font-weight:700;font-size:14px">{camp_id}</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    _metric_box("Avg Perf Score", f"{avg_perf:.3f}", purple=True),
                    unsafe_allow_html=True,
                )
                st.markdown(_metric_box("Objective", str(objective)), unsafe_allow_html=True)
                st.markdown(_metric_box("KPI Goal", str(kpi_goal)), unsafe_allow_html=True)
                st.markdown(_metric_box("Creatives", str(n_creatives)), unsafe_allow_html=True)
                if st.button(
                    "View Campaign →", key=f"camp_btn_{camp_id}", use_container_width=True
                ):
                    st.session_state.current_view = "campaign"
                    st.session_state.selected_campaign = camp_id
                    logger.debug("Navigating to campaign view: {}", camp_id)
                    st.rerun()
