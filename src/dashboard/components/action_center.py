"""Action Center — prioritised portfolio intelligence for the overview page."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from loguru import logger

_PURPLE = "#881CA6"
_RED = "#dc2626"
_GREEN = "#059669"
_ORANGE = "#d97706"

_BINARY_ATTRS = ["has_gameplay", "has_ugc_style", "has_price", "has_discount_badge"]
_CAT_ATTRS = ["theme", "emotional_tone", "hook_type", "dominant_color"]

_ATTR_LABELS = {
    "has_gameplay": "gameplay elements",
    "has_ugc_style": "UGC-style content",
    "has_price": "price callouts",
    "has_discount_badge": "discount badges",
    "theme": "theme",
    "emotional_tone": "emotional tone",
    "hook_type": "hook type",
    "dominant_color": "dominant colour",
}


def _card(border_color: str, header: str, body_html: str) -> None:
    st.markdown(
        f'<div style="border-left:4px solid {border_color};background:rgba(255,255,255,0.55);'
        f"border-radius:0 12px 12px 0;padding:14px 16px;margin-bottom:10px;"
        f'box-shadow:0 1px 4px rgba(0,0,0,0.06)">'
        f'<div style="font-size:11px;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.07em;color:{border_color};margin-bottom:6px">{header}</div>'
        f"{body_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


def _navigate_to_ad(creative_id: str, campaign_id: str) -> None:
    """Set session state to navigate to the ad detail view for the given creative."""
    st.session_state.selected_creative = creative_id
    st.session_state.selected_campaign = campaign_id
    st.session_state.current_view = "ad_detail"


def _pause_now_column(your_summary: pd.DataFrame) -> None:
    """Render the 'Pause Now' column listing the top fatigued creatives by wasted spend."""
    st.markdown(
        f'<div style="font-size:14px;font-weight:800;color:{_RED};margin-bottom:10px">'
        f"🚨 Pause Now</div>",
        unsafe_allow_html=True,
    )

    fatigued = your_summary[your_summary["creative_status"] == "fatigued"].copy()

    if fatigued.empty:
        st.markdown(
            '<div style="color:#888;font-size:13px;padding:8px 0">'
            "No fatigued creatives — portfolio looks healthy!</div>",
            unsafe_allow_html=True,
        )
        return

    # days fatigued = total_days_active - fatigue_day
    fatigued["days_fatigued"] = (fatigued["total_days_active"] - fatigued["fatigue_day"]).clip(
        lower=0
    )

    # daily burn rate * days since fatigue = estimated wasted spend
    daily_rate = fatigued["total_spend_usd"] / fatigued["total_days_active"].replace(0, 1)
    fatigued["wasted_usd"] = (daily_rate * fatigued["days_fatigued"]).fillna(0)

    fatigued = fatigued.sort_values("days_fatigued", ascending=False).head(3)

    for _, row in fatigued.iterrows():
        days = int(row["days_fatigued"])
        wasted = row["wasted_usd"]
        cid = row["creative_id"]
        camp = row["campaign_id"]

        body = (
            f'<div style="font-size:18px;font-weight:800;color:{_RED};line-height:1">{cid}</div>'
            f'<div style="font-size:12px;color:#555;margin-top:2px">{camp}</div>'
            f'<div style="font-size:12px;color:{_RED};margin-top:6px">'
            f"Fatigued {days} day{'s' if days != 1 else ''} ago · "
            f"~${wasted:,.0f} wasted spend</div>"
        )
        _card(_RED, "⚠ Fatigued Creative", body)

        if st.button(f"View {cid}", key=f"pause_{cid}"):
            _navigate_to_ad(cid, camp)
            st.rerun()


def _scale_these_column(your_summary: pd.DataFrame) -> None:
    st.markdown(
        f'<div style="font-size:14px;font-weight:800;color:{_GREEN};margin-bottom:10px">'
        f"⚡ Scale These</div>",
        unsafe_allow_html=True,
    )

    top = your_summary[your_summary["creative_status"] == "top_performer"].copy()
    top = top.sort_values("overall_roas", ascending=False).head(3)

    if top.empty:
        st.markdown(
            '<div style="color:#888;font-size:13px;padding:8px 0">'
            "No top performers yet — keep running campaigns to gather data.</div>",
            unsafe_allow_html=True,
        )
        return

    for _, row in top.iterrows():
        cid = row["creative_id"]
        camp = row["campaign_id"]
        roas = row["overall_roas"]
        perf = row["perf_score"]

        body = (
            f'<div style="font-size:18px;font-weight:800;color:{_GREEN};line-height:1">{cid}</div>'
            f'<div style="font-size:12px;color:#555;margin-top:2px">{camp}</div>'
            f'<div style="display:flex;gap:10px;margin-top:6px">'
            f'<span style="font-size:12px;background:#dcfce7;color:{_GREEN};'
            f'padding:2px 8px;border-radius:20px;font-weight:700">ROAS {roas:.2f}×</span>'
            f'<span style="font-size:12px;background:#f3e8ff;color:{_PURPLE};'
            f'padding:2px 8px;border-radius:20px;font-weight:700">Score {perf:.3f}</span>'
            f"</div>"
        )
        _card(_GREEN, "★ Top Performer", body)

        if st.button(f"View {cid}", key=f"scale_{cid}"):
            _navigate_to_ad(cid, camp)
            st.rerun()


def _test_next_column(your_summary: pd.DataFrame) -> None:
    st.markdown(
        f'<div style="font-size:14px;font-weight:800;color:{_PURPLE};margin-bottom:10px">'
        f"🧪 Test Next</div>",
        unsafe_allow_html=True,
    )

    top = your_summary[your_summary["creative_status"] == "top_performer"]
    all_df = your_summary

    if top.empty or len(top) < 2:
        st.markdown(
            '<div style="color:#888;font-size:13px;padding:8px 0">'
            "Need more top performers to identify winning patterns.</div>",
            unsafe_allow_html=True,
        )
        return

    insights: list[tuple[float, str]] = []

    # Binary attributes
    for attr in _BINARY_ATTRS:
        if attr not in all_df.columns:
            continue
        top_rate = float(top[attr].fillna(0).mean())
        all_rate = float(all_df[attr].fillna(0).mean())
        lift = top_rate - all_rate
        if top_rate > 0.35 and lift > 0.05:
            pct = int(round(top_rate * 100))
            label = _ATTR_LABELS.get(attr, attr)
            insights.append((lift, f"{pct}% of top performers use {label}"))

    # Categorical attributes — find dominant value in top performers
    for attr in _CAT_ATTRS:
        if attr not in all_df.columns:
            continue
        top_counts = top[attr].dropna().value_counts(normalize=True)
        all_counts = all_df[attr].dropna().value_counts(normalize=True)
        if top_counts.empty:
            continue
        best_val = top_counts.index[0]
        top_rate = float(top_counts.iloc[0])
        all_rate = float(all_counts.get(best_val, 0))
        lift = top_rate - all_rate
        if top_rate > 0.30 and lift > 0.05:
            pct = int(round(top_rate * 100))
            label = _ATTR_LABELS.get(attr, attr)
            insights.append((lift, f"{pct}% of top performers: {label} = '{best_val}'"))

    insights.sort(reverse=True)
    insights = insights[:4]

    if not insights:
        # Fallback: show top attributes by frequency regardless of lift
        for attr in _BINARY_ATTRS:
            if attr in top.columns:
                rate = float(top[attr].fillna(0).mean())
                if rate > 0.4:
                    label = _ATTR_LABELS.get(attr, attr)
                    insights.append((rate, f"{int(rate * 100)}% of top performers use {label}"))
        insights.sort(reverse=True)
        insights = insights[:3]

    bullets = "".join(
        f'<li style="margin-bottom:6px;font-size:13px">{msg}</li>' for _, msg in insights
    )

    n_top = len(top)
    n_all = len(all_df)

    body = (
        f'<div style="font-size:12px;color:#666;margin-bottom:8px">'
        f"Patterns from {n_top}/{n_all} top-performing creatives:</div>"
        f'<ul style="margin:0;padding-left:18px;color:#333">{bullets}</ul>'
        f'<div style="margin-top:10px;font-size:12px;color:{_PURPLE};font-weight:600">'
        f"→ Replicate these in your next creative brief</div>"
    )
    _card(_PURPLE, "✦ Winning Formula", body)


def render_action_center(your_summary: pd.DataFrame) -> None:
    """Render the three-column action center panel."""
    logger.debug("Rendering action center for {} creatives", len(your_summary))

    st.markdown(
        '<div style="font-size:20px;font-weight:800;color:#1a1a2e;margin-bottom:4px">'
        "Action Center</div>"
        '<div style="font-size:13px;color:#666;margin-bottom:16px">'
        "Prioritised recommendations based on your creative portfolio</div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        _pause_now_column(your_summary)

    with col2:
        _scale_these_column(your_summary)

    with col3:
        _test_next_column(your_summary)
