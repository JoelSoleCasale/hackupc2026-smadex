"""Campaign cards component for the Smadex Creative Intelligence dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from loguru import logger


def _perf_badge(score: float) -> str:
    if score >= 0.6:
        return f"🟢 {score:.3f}"
    elif score >= 0.4:
        return f"🟡 {score:.3f}"
    else:
        return f"🔴 {score:.3f}"


def render_campaign_cards(adv_summary: pd.DataFrame, adv_campaigns: pd.DataFrame) -> None:
    """Render 5 clickable campaign cards in a row.

    Parameters
    ----------
    adv_summary:
        creative_summary rows for the selected advertiser (already ID-mapped).
    adv_campaigns:
        campaigns.csv rows for the selected advertiser (already ID-mapped).
    """
    campaigns = sorted(adv_summary["campaign_id"].unique())
    logger.debug("render_campaign_cards: {} campaigns", len(campaigns))

    if not campaigns:
        st.info("No campaigns found for this advertiser.")
        return

    st.subheader("Campaigns")
    cols = st.columns(len(campaigns))

    for col, camp_id in zip(cols, campaigns):
        camp_summary = adv_summary[adv_summary["campaign_id"] == camp_id]
        avg_perf = float(camp_summary["perf_score"].mean()) if not camp_summary.empty else 0.0

        # Get metadata from campaigns df
        camp_meta = adv_campaigns[adv_campaigns["campaign_id"] == camp_id]
        objective = camp_meta["objective"].iloc[0] if not camp_meta.empty else "—"
        kpi_goal = camp_meta["kpi_goal"].iloc[0] if not camp_meta.empty else "—"
        n_creatives = len(camp_summary)

        with col:
            with st.container(border=True):
                st.markdown(f"**{camp_id}**")
                st.markdown(_perf_badge(avg_perf))
                st.caption(f"Objective: {objective}")
                st.caption(f"KPI: {kpi_goal}")
                st.caption(f"{n_creatives} creatives")
                if st.button(
                    "View Campaign →", key=f"camp_btn_{camp_id}", use_container_width=True
                ):
                    st.session_state.current_view = "campaign"
                    st.session_state.selected_campaign = camp_id
                    logger.debug("Navigating to campaign view: {}", camp_id)
                    st.rerun()
