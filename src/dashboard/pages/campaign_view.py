"""Campaign detail view for the Smadex Creative Intelligence dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from loguru import logger

from src.dashboard.components.creative_grid import render_creative_grid
from src.dashboard.components.kpi_cards import render_kpi_cards


def render_campaign_view(
    summary_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    campaigns_df: pd.DataFrame,
    peers_summary: pd.DataFrame,
    advertiser: str,
    campaign_id: str,
) -> None:
    """Render the campaign detail page.

    Parameters
    ----------
    summary_df:
        Full creative_summary (all advertisers, ID-mapped).
    daily_df:
        Full daily stats (all advertisers, ID-mapped).
    campaigns_df:
        Campaigns metadata (all advertisers, ID-mapped).
    peers_summary:
        creative_summary rows for sector peers (same vertical, different advertiser).
    advertiser:
        Selected advertiser name.
    campaign_id:
        Selected campaign ID (e.g. "Campaign 1").
    """
    # Back navigation
    if st.button("← Back to Overview"):
        st.session_state.current_view = "overview"
        st.session_state.selected_campaign = None
        st.rerun()

    st.markdown(f"**{advertiser}** › {campaign_id}")
    st.title(campaign_id)

    # Filter data for this campaign
    camp_summary = summary_df[
        (summary_df["advertiser_name"] == advertiser) & (summary_df["campaign_id"] == campaign_id)
    ]
    camp_creative_ids = camp_summary["creative_id"].tolist()
    camp_daily = daily_df[daily_df["creative_id"].isin(camp_creative_ids)]

    logger.debug(
        "campaign_view: advertiser={} campaign={} creatives={} daily_rows={}",
        advertiser,
        campaign_id,
        len(camp_summary),
        len(camp_daily),
    )

    # KPI cards
    render_kpi_cards(camp_summary, camp_daily, {"metric": "perf_score"}, peers_summary)

    st.divider()

    # Campaign metadata
    camp_meta_rows = campaigns_df[
        (campaigns_df["advertiser_name"] == advertiser)
        & (campaigns_df["campaign_id"] == campaign_id)
    ]

    if not camp_meta_rows.empty:
        meta = camp_meta_rows.iloc[0]
        st.subheader("Campaign Details")
        left, right = st.columns(2)

        with left:
            st.markdown(f"**Objective:** {meta.get('objective', '—')}")
            st.markdown(f"**KPI Goal:** {meta.get('kpi_goal', '—')}")
            st.markdown(f"**Primary Theme:** {meta.get('primary_theme', '—')}")
            budget = meta.get("daily_budget_usd", None)
            budget_str = f"${float(budget):,.0f}/day" if pd.notna(budget) else "—"
            st.markdown(f"**Daily Budget:** {budget_str}")

        with right:
            st.markdown(f"**Target Age:** {meta.get('target_age_segment', '—')}")
            st.markdown(f"**Target OS:** {meta.get('target_os', '—')}")
            countries_raw = meta.get("countries", "")
            if pd.notna(countries_raw) and countries_raw:
                countries_list = str(countries_raw).replace(";", ",").replace("|", ",")
                st.markdown(f"**Countries:** {countries_list}")
            else:
                st.markdown("**Countries:** —")
            start = meta.get("start_date", "—")
            end = meta.get("end_date", "—")
            st.markdown(f"**Period:** {start} → {end}")

    st.divider()

    render_creative_grid(camp_summary)
