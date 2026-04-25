"""Ad detail view for the Smadex Creative Intelligence dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from loguru import logger

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

_BAD_STATUSES = {"underperformer", "fatigued"}

_ATTRIBUTE_FIELDS = [
    ("format", "Format"),
    ("theme", "Theme"),
    ("emotional_tone", "Emotional Tone"),
    ("language", "Language"),
    ("hook_type", "Hook Type"),
    ("cta_text", "CTA Text"),
    ("headline", "Headline"),
    ("duration_sec", "Duration (s)"),
    ("width", "Width"),
    ("height", "Height"),
    ("has_gameplay", "Has Gameplay"),
    ("has_ugc_style", "Has UGC Style"),
    ("has_price", "Has Price"),
    ("has_discount_badge", "Has Discount Badge"),
]


def _asset_path(asset_file: str) -> Path:
    return _PROJECT_ROOT / "data" / asset_file


def render_ad_detail_view(
    summary_df: pd.DataFrame,
    advertiser: str,
    campaign_id: str,
    creative_id: str,
) -> None:
    """Render the ad detail page.

    Parameters
    ----------
    summary_df:
        Full creative_summary (all advertisers, ID-mapped).
    advertiser:
        Selected advertiser name.
    campaign_id:
        Selected campaign ID (e.g. "Campaign 1").
    creative_id:
        Selected creative ID (e.g. "Creative 1.3").
    """
    # Back navigation
    if st.button("← Back to Campaign"):
        st.session_state.current_view = "campaign"
        st.session_state.selected_creative = None
        st.rerun()

    st.markdown(f"**{advertiser}** › {campaign_id} › {creative_id}")
    st.title(creative_id)

    # Get the creative row — filter by both creative_id and advertiser to avoid
    # collisions (all advertisers share the same mapped IDs like "Creative 1.1")
    creative_rows = summary_df[
        (summary_df["creative_id"] == creative_id) & (summary_df["advertiser_name"] == advertiser)
    ]
    if creative_rows.empty:
        st.warning("Creative not found.")
        logger.warning("ad_detail_view: creative_id={} not found in summary_df", creative_id)
        return

    row = creative_rows.iloc[0]
    status = str(row.get("creative_status", "stable"))
    asset_file = row.get("asset_file", "")
    img_path = _asset_path(asset_file) if asset_file else Path("/nonexistent")

    logger.debug(
        "ad_detail_view: creative={} status={} perf_score={}",
        creative_id,
        status,
        row.get("perf_score"),
    )

    left, right = st.columns([2, 3])

    with left:
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
        else:
            st.markdown(
                "<div style='height:300px;background:#eee;display:flex;"
                "align-items:center;justify-content:center;border-radius:8px'>"
                "No image available</div>",
                unsafe_allow_html=True,
            )
        status_emoji = "😊" if status not in _BAD_STATUSES else "😔"
        st.markdown(f"{status_emoji} **{status.replace('_', ' ').title()}**")

    with right:
        # --- Metrics ---
        st.subheader("Metrics")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            perf = row.get("perf_score", None)
            st.metric("Perf Score", f"{float(perf):.3f}" if pd.notna(perf) else "N/A")
        with m2:
            ctr = row.get("overall_ctr", None)
            st.metric("CTR", f"{float(ctr) * 100:.2f}%" if pd.notna(ctr) else "N/A")
        with m3:
            cvr = row.get("overall_cvr", None)
            st.metric("CVR", f"{float(cvr) * 100:.2f}%" if pd.notna(cvr) else "N/A")
        with m4:
            roas = row.get("overall_roas", None)
            st.metric("ROAS", f"{float(roas):.2f}×" if pd.notna(roas) else "N/A")

        st.divider()

        # --- Attributes ---
        st.subheader("Attributes")
        attr_col1, attr_col2 = st.columns(2)
        fields_left = _ATTRIBUTE_FIELDS[: len(_ATTRIBUTE_FIELDS) // 2 + 1]
        fields_right = _ATTRIBUTE_FIELDS[len(_ATTRIBUTE_FIELDS) // 2 + 1 :]

        with attr_col1:
            for col_key, label in fields_left:
                val = row.get(col_key, None)
                if val is not None and pd.notna(val):
                    st.markdown(f"**{label}:** {val}")

        with attr_col2:
            for col_key, label in fields_right:
                val = row.get(col_key, None)
                if val is not None and pd.notna(val):
                    st.markdown(f"**{label}:** {val}")

        st.divider()

        # --- Performance Explainability ---
        st.subheader("Performance Explainability")
        st.info(
            "🔬 Performance explainability coming soon — we will explain why this creative "
            "performs well or poorly based on visual and contextual signals."
        )

        # --- See Alternatives (bad performers only) ---
        if status in _BAD_STATUSES:
            st.divider()
            st.subheader("Alternatives")
            st.warning(
                "This creative is underperforming. You can explore similar creatives that "
                "perform better in this segment."
            )
            if st.button("🔄 See Alternatives (coming soon)", use_container_width=True):
                st.info("Alternative recommendations are not yet implemented.")
