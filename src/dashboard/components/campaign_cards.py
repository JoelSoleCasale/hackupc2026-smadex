"""Campaign cards component for the Smadex Creative Intelligence dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from loguru import logger

_PURPLE = "#881CA6"


def _perf_badge(score: float) -> str:
    if score >= 0.6:
        return (
            '<span style="background:#f0fdf4;color:#16a34a;border:1px solid #bbf7d0;'
            f'border-radius:20px;padding:2px 10px;font-size:12px;font-weight:700">🟢 {score:.3f}</span>'
        )
    elif score >= 0.4:
        return (
            '<span style="background:#fffbeb;color:#d97706;border:1px solid #fde68a;'
            f'border-radius:20px;padding:2px 10px;font-size:12px;font-weight:700">🟡 {score:.3f}</span>'
        )
    else:
        return (
            '<span style="background:#fef2f2;color:#dc2626;border:1px solid #fecaca;'
            f'border-radius:20px;padding:2px 10px;font-size:12px;font-weight:700">🔴 {score:.3f}</span>'
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
                st.markdown(_perf_badge(avg_perf), unsafe_allow_html=True)
                st.write("")
                st.caption(f"**Objective:** {objective}")
                st.caption(f"**KPI:** {kpi_goal}")
                st.caption(f"**Creatives:** {n_creatives}")
                if st.button(
                    "View Campaign →", key=f"camp_btn_{camp_id}", use_container_width=True
                ):
                    st.session_state.current_view = "campaign"
                    st.session_state.selected_campaign = camp_id
                    logger.debug("Navigating to campaign view: {}", camp_id)
                    st.rerun()
