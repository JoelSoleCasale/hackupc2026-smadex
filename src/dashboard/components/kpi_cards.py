"""KPI cards row for the Smadex Creative Intelligence dashboard."""

from __future__ import annotations

import math

import pandas as pd
import scipy.stats
import streamlit as st
from loguru import logger

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_divide(numerator: float, denominator: float) -> float:
    """Return numerator / denominator, or NaN if denominator is 0."""
    if isinstance(numerator, float) and math.isnan(numerator):
        return float("nan")
    if denominator == 0 or math.isnan(denominator):
        return float("nan")
    return numerator / denominator


def _percentile_caption(peers_series: pd.Series, your_val: float, sector: str) -> None:
    """Render a st.caption showing percentile rank within peers."""
    clean = peers_series.dropna()
    if clean.empty or math.isnan(your_val):
        st.caption(f"No peer data for {sector}")
        return
    pct = scipy.stats.percentileofscore(clean, your_val, kind="rank")
    st.caption(f"Top {100 - pct:.0f}% of {sector}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_kpi_cards(
    your_summary: pd.DataFrame,
    your_daily: pd.DataFrame,
    filters: dict,
    peers_summary: pd.DataFrame,
) -> None:
    """Render a row of 5 KPI cards comparing your advertiser to sector peers.

    Parameters
    ----------
    your_summary:
        Filtered ``creative_summary`` rows for the selected advertiser.
    your_daily:
        Filtered ``creative_daily_country_os_stats`` rows for the selected
        advertiser (pre-joined with campaign/creative metadata by the loader).
    filters:
        Filter state dict returned by ``render_filters()``.
    peers_summary:
        Rows from ``creative_summary`` belonging to other advertisers in the
        same sector/vertical as the selected advertiser.
    """
    logger.debug("Rendering KPI cards")

    # ------------------------------------------------------------------
    # Determine sector label
    # ------------------------------------------------------------------
    if not your_summary.empty and "vertical" in your_summary.columns:
        sector = str(your_summary["vertical"].iloc[0])
    else:
        sector = "sector"

    # ------------------------------------------------------------------
    # Compute "your" metrics
    # ------------------------------------------------------------------
    summary_empty = your_summary.empty
    daily_empty = your_daily.empty

    # Perf score (from summary)
    if summary_empty:
        your_perf: float | None = None
    else:
        your_perf = float(your_summary["perf_score"].median())

    # CTR, CVR, ROAS, Spend (aggregated from daily)
    if daily_empty:
        your_ctr = your_cvr = your_roas = your_spend = None
    else:
        total_impressions = your_daily["impressions"].sum()
        total_clicks = your_daily["clicks"].sum()
        total_conversions = your_daily["conversions"].sum()
        total_revenue = your_daily["revenue_usd"].sum()
        total_spend = your_daily["spend_usd"].sum()

        your_ctr = _safe_divide(total_clicks, total_impressions)
        your_cvr = _safe_divide(total_conversions, total_clicks)
        your_roas = _safe_divide(total_revenue, total_spend)
        your_spend = total_spend

    # ------------------------------------------------------------------
    # Compute peer medians
    # ------------------------------------------------------------------
    peers_empty = peers_summary.empty

    peer_perf = float(peers_summary["perf_score"].median()) if not peers_empty else float("nan")
    peer_ctr = float(peers_summary["overall_ctr"].median()) if not peers_empty else float("nan")
    peer_cvr = float(peers_summary["overall_cvr"].median()) if not peers_empty else float("nan")
    peer_roas = float(peers_summary["overall_roas"].median()) if not peers_empty else float("nan")

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------
    cols = st.columns(5)

    # ---- Card 1: Perf Score ----
    with cols[0]:
        if your_perf is None or math.isnan(your_perf):
            st.metric("Perf Score", "N/A")
        else:
            delta_val = your_perf - peer_perf if not math.isnan(peer_perf) else None
            delta_str = f"{delta_val:+.3f}" if delta_val is not None else None
            st.metric(
                label="Perf Score",
                value=f"{your_perf:.3f}",
                delta=delta_str,
            )
            if not peers_empty:
                _percentile_caption(peers_summary["perf_score"], your_perf, sector)

    # ---- Card 2: CTR ----
    with cols[1]:
        if your_ctr is None or math.isnan(your_ctr):
            st.metric("CTR", "N/A")
        else:
            delta_val = (your_ctr - peer_ctr) if not math.isnan(peer_ctr) else None
            delta_str = f"{delta_val * 100:+.2f}pp" if delta_val is not None else None
            st.metric(
                label="CTR",
                value=f"{your_ctr * 100:.2f}%",
                delta=delta_str,
            )
            if not peers_empty:
                _percentile_caption(peers_summary["overall_ctr"], your_ctr, sector)

    # ---- Card 3: CVR ----
    with cols[2]:
        if your_cvr is None or math.isnan(your_cvr):
            st.metric("CVR", "N/A")
        else:
            delta_val = (your_cvr - peer_cvr) if not math.isnan(peer_cvr) else None
            delta_str = f"{delta_val * 100:+.2f}pp" if delta_val is not None else None
            st.metric(
                label="CVR",
                value=f"{your_cvr * 100:.2f}%",
                delta=delta_str,
            )
            if not peers_empty:
                _percentile_caption(peers_summary["overall_cvr"], your_cvr, sector)

    # ---- Card 4: ROAS ----
    with cols[3]:
        if your_roas is None or math.isnan(your_roas):
            st.metric("ROAS", "N/A")
        else:
            delta_val = (your_roas - peer_roas) if not math.isnan(peer_roas) else None
            delta_str = f"{delta_val:+.3f}" if delta_val is not None else None
            st.metric(
                label="ROAS",
                value=f"{your_roas:.2f}×",
                delta=delta_str,
            )
            if not peers_empty:
                _percentile_caption(peers_summary["overall_roas"], your_roas, sector)

    # ---- Card 5: Spend ----
    with cols[4]:
        if your_spend is None:
            st.metric("Spend", "N/A")
        else:
            st.metric(
                label="Spend",
                value=f"${your_spend:,.0f}",
            )

    logger.debug(
        "KPI cards rendered — perf={}, ctr={}, cvr={}, roas={}, spend={}",
        your_perf,
        your_ctr,
        your_cvr,
        your_roas,
        your_spend,
    )
