"""KPI cards row for the Smadex Creative Intelligence dashboard."""

from __future__ import annotations

import math

import pandas as pd
import scipy.stats
import streamlit as st
from loguru import logger

_PURPLE = "#881CA6"
_PURPLE_DARK = "#6a1580"

# Per-metric accent colour, icon, label
_METRIC_STYLE = {
    "ctr": ("#2563eb", "📊", "CTR"),
    "cvr": ("#059669", "🔄", "CVR"),
    "roas": ("#d97706", "💰", "ROAS"),
    "spend": ("#64748b", "💸", "Spend"),
}


def _safe_divide(numerator: float, denominator: float) -> float:
    if isinstance(numerator, float) and math.isnan(numerator):
        return float("nan")
    if denominator == 0 or math.isnan(denominator):
        return float("nan")
    return numerator / denominator


def _percentile_rank(peers_series: pd.Series, your_val: float) -> float | None:
    clean = peers_series.dropna()
    if clean.empty or math.isnan(your_val):
        return None
    return scipy.stats.percentileofscore(clean, your_val, kind="rank")


def _perf_score_card(
    value: float, delta: float | None, percentile: float | None, sector: str
) -> None:
    """Prominent purple gradient card for the Perf Score metric."""
    delta_html = ""
    if delta is not None:
        sign = "+" if delta >= 0 else ""
        color = "rgba(255,255,255,0.95)" if delta >= 0 else "#ffb3b3"
        delta_html = (
            f'<div style="font-size:13px;color:{color};margin-top:2px">'
            f"{sign}{delta:.3f} vs peers</div>"
        )
    pct_html = ""
    if percentile is not None:
        pct_html = (
            f'<div style="font-size:11px;color:rgba(255,255,255,0.7);margin-top:4px">'
            f"Top {100 - percentile:.0f}% of {sector}</div>"
        )
    st.markdown(
        f'<div style="background:linear-gradient(135deg,{_PURPLE},{_PURPLE_DARK});'
        f'border-radius:14px;padding:20px 22px;height:100%">'
        f'<div style="font-size:11px;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.08em;color:rgba(255,255,255,0.75);margin-bottom:6px">⭐ Perf Score</div>'
        f'<div style="font-size:34px;font-weight:800;color:#fff;line-height:1">{value:.3f}</div>'
        f"{delta_html}{pct_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


def _accent_metric(
    key: str, value_str: str, delta_str: str | None, percentile: float | None, sector: str
) -> None:
    """Metric card with a coloured left border accent, icon, and percentile bar."""
    color, icon, label = _METRIC_STYLE[key]

    delta_html = ""
    if delta_str is not None:
        is_pos = delta_str.startswith("+")
        d_color = "#16a34a" if is_pos else "#dc2626"
        delta_html = (
            f'<div style="font-size:12px;color:{d_color};margin-top:2px">{delta_str}</div>'
        )

    bar_html = ""
    if percentile is not None:
        fill = max(2, min(100, 100 - percentile))
        bar_html = (
            f'<div style="margin-top:8px">'
            f'<div style="font-size:10px;color:#888;margin-bottom:2px">Top {100 - percentile:.0f}% of {sector}</div>'
            f'<div style="height:4px;background:#e5e7eb;border-radius:2px">'
            f'<div style="height:4px;width:{fill:.0f}%;background:{color};border-radius:2px"></div>'
            f"</div></div>"
        )

    st.markdown(
        f'<div style="border-left:4px solid {color};padding:12px 14px;border-radius:0 10px 10px 0">'
        f'<div style="font-size:11px;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.06em;color:#888;margin-bottom:4px">{icon} {label}</div>'
        f'<div style="font-size:26px;font-weight:800;color:{color};line-height:1">{value_str}</div>'
        f"{delta_html}{bar_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_kpi_cards(
    your_summary: pd.DataFrame,
    your_daily: pd.DataFrame,
    filters: dict,
    peers_summary: pd.DataFrame,
) -> None:
    logger.debug("Rendering KPI cards")

    sector = (
        str(your_summary["vertical"].iloc[0])
        if not your_summary.empty and "vertical" in your_summary.columns
        else "sector"
    )

    your_perf = float(your_summary["perf_score"].median()) if not your_summary.empty else None

    if your_daily.empty:
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

    peers_empty = peers_summary.empty
    peer_perf = float(peers_summary["perf_score"].median()) if not peers_empty else float("nan")
    peer_ctr = float(peers_summary["overall_ctr"].median()) if not peers_empty else float("nan")
    peer_cvr = float(peers_summary["overall_cvr"].median()) if not peers_empty else float("nan")
    peer_roas = float(peers_summary["overall_roas"].median()) if not peers_empty else float("nan")

    cols = st.columns(5)

    # ── Perf Score (purple card) ────────────────────────────────────────
    with cols[0]:
        if your_perf is None or math.isnan(your_perf):
            st.metric("Perf Score", "N/A")
        else:
            delta = your_perf - peer_perf if not math.isnan(peer_perf) else None
            pct = (
                _percentile_rank(peers_summary["perf_score"], your_perf)
                if not peers_empty
                else None
            )
            _perf_score_card(your_perf, delta, pct, sector)

    # ── CTR ────────────────────────────────────────────────────────────
    with cols[1]:
        if your_ctr is None or math.isnan(your_ctr):
            st.metric("CTR", "N/A")
        else:
            delta = your_ctr - peer_ctr if not math.isnan(peer_ctr) else None
            delta_str = f"{delta * 100:+.2f}pp" if delta is not None else None
            pct = (
                _percentile_rank(peers_summary["overall_ctr"], your_ctr)
                if not peers_empty
                else None
            )
            _accent_metric("ctr", f"{your_ctr * 100:.2f}%", delta_str, pct, sector)

    # ── CVR ────────────────────────────────────────────────────────────
    with cols[2]:
        if your_cvr is None or math.isnan(your_cvr):
            st.metric("CVR", "N/A")
        else:
            delta = your_cvr - peer_cvr if not math.isnan(peer_cvr) else None
            delta_str = f"{delta * 100:+.2f}pp" if delta is not None else None
            pct = (
                _percentile_rank(peers_summary["overall_cvr"], your_cvr)
                if not peers_empty
                else None
            )
            _accent_metric("cvr", f"{your_cvr * 100:.2f}%", delta_str, pct, sector)

    # ── ROAS ───────────────────────────────────────────────────────────
    with cols[3]:
        if your_roas is None or math.isnan(your_roas):
            st.metric("ROAS", "N/A")
        else:
            delta = your_roas - peer_roas if not math.isnan(peer_roas) else None
            delta_str = f"{delta:+.3f}" if delta is not None else None
            pct = (
                _percentile_rank(peers_summary["overall_roas"], your_roas)
                if not peers_empty
                else None
            )
            _accent_metric("roas", f"{your_roas:.2f}×", delta_str, pct, sector)

    # ── Spend ──────────────────────────────────────────────────────────
    with cols[4]:
        if your_spend is None:
            st.metric("Spend", "N/A")
        else:
            _accent_metric("spend", f"${your_spend:,.0f}", None, None, sector)

    logger.debug(
        "KPI cards rendered — perf={}, ctr={}, cvr={}, roas={}, spend={}",
        your_perf,
        your_ctr,
        your_cvr,
        your_roas,
        your_spend,
    )
