"""Campaign detail view for the Smadex Creative Intelligence dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from loguru import logger

from src.dashboard.components.creative_grid import render_creative_grid
from src.dashboard.components.kpi_cards import render_kpi_cards


def _render_roas_evolution(camp_daily: pd.DataFrame, camp_summary: pd.DataFrame) -> go.Figure:
    """Multi-line ROAS over time, one line per creative."""
    if camp_daily.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No daily data",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 14, "color": "grey"},
        )
        fig.update_layout(xaxis_visible=False, yaxis_visible=False, height=420)
        return fig

    daily = camp_daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    agg = (
        daily.groupby(["date", "creative_id"])
        .agg(revenue=("revenue_usd", "sum"), spend=("spend_usd", "sum"))
        .reset_index()
    )
    agg["roas"] = agg["revenue"] / agg["spend"].replace(0, float("nan"))

    status_map = camp_summary.set_index("creative_id")["creative_status"].to_dict()

    fig = go.Figure()
    for cid in sorted(agg["creative_id"].unique()):
        cdata = agg[agg["creative_id"] == cid].sort_values("date")
        status = status_map.get(cid, "stable")
        dash = "dot" if status in ("fatigued", "underperformer") else "solid"
        fig.add_trace(
            go.Scatter(
                x=cdata["date"],
                y=cdata["roas"],
                mode="lines",
                name=cid,
                line={"dash": dash},
                hovertemplate="%{fullData.name}<br>%{x|%b %d}<br>ROAS: %{y:.2f}×<extra></extra>",
            )
        )

    fig.update_layout(
        title="ROAS Evolution",
        xaxis_title="Date",
        yaxis_title="ROAS",
        height=420,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig


def _render_impression_share(camp_daily: pd.DataFrame) -> go.Figure:
    """Stacked area chart of daily impressions per creative."""
    if camp_daily.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No daily data",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 14, "color": "grey"},
        )
        fig.update_layout(xaxis_visible=False, yaxis_visible=False, height=420)
        return fig

    daily = camp_daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    agg = daily.groupby(["date", "creative_id"])["impressions"].sum().reset_index()

    fig = go.Figure()
    for cid in sorted(agg["creative_id"].unique()):
        cdata = agg[agg["creative_id"] == cid].sort_values("date")
        fig.add_trace(
            go.Scatter(
                x=cdata["date"],
                y=cdata["impressions"],
                mode="lines",
                name=cid,
                stackgroup="one",
                hovertemplate="%{fullData.name}<br>%{x|%b %d}<br>Impressions: %{y:,.0f}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Impression Share Over Time",
        xaxis_title="Date",
        yaxis_title="Impressions",
        height=420,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig


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

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(_render_roas_evolution(camp_daily, camp_summary), use_container_width=True)
    with col2:
        st.plotly_chart(_render_impression_share(camp_daily), use_container_width=True)

    st.divider()

    render_creative_grid(camp_summary)

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
