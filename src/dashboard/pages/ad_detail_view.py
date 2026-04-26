"""Ad detail view for the Smadex Creative Intelligence dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from loguru import logger

from src.analysis.fitness_scorer import (
    LinearFitnessScorer,
    SharpeCorrelationScorer,
    TopKFitnessScorer,
)
from src.data.loader import load_correlations

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


@st.cache_data
def _cached_fitness_scores(
    summary_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    target_segments: tuple[tuple[str, str], ...],
    scorer_name: str,
) -> pd.Series:
    scorer_map = {
        "Linear": LinearFitnessScorer(),
        "Sharpe": SharpeCorrelationScorer(),
        "Top-10": TopKFitnessScorer(k=10),
    }
    return scorer_map[scorer_name].score_all(summary_df, corr_df, list(target_segments))


_BAD_STATUSES = {"underperformer", "fatigued"}

_ATTRIBUTE_FIELDS = [
    ("format", "Format"),
    ("theme", "Theme"),
    ("emotional_tone", "Emotional Tone"),
    ("language", "Language"),
    ("hook_type", "Hook Type"),
    ("duration_sec", "Duration (s)"),
    ("has_gameplay", "Has Gameplay"),
    ("has_ugc_style", "Has UGC Style"),
    ("has_price", "Has Price"),
    ("has_discount_badge", "Has Discount Badge"),
]

_SCORERS = [
    (
        "Linear",
        LinearFitnessScorer(),
        "maximises total alignment across all audience signals",
    ),
    (
        "Sharpe",
        SharpeCorrelationScorer(),
        "rewards creatives with consistent alignment across attributes",
    ),
    (
        "Top-10",
        TopKFitnessScorer(k=10),
        "focuses on the 10 strongest audience signals",
    ),
]


def _asset_path(asset_file: str) -> Path:
    return _PROJECT_ROOT / "data" / asset_file


def _explainability_tags(
    corr_df: pd.DataFrame,
    attribute: str,
    value: object,
    threshold: float = 0.01,
) -> list[tuple[str, str]]:
    """Return grouped (emoji, 'characteristic: val1, val2') pairs for significant correlations.

    Rows sharing the same user_characteristic and correlation sign are merged into a
    single tag so that e.g. country:US and country:UK become 'country: US, UK'.
    Returns an empty list when the attribute is absent or has no significant rows.
    """
    attr_rows = corr_df[corr_df["creative_attribute"] == attribute]
    if attr_rows.empty:
        return []

    is_categorical = attr_rows["creative_attribute_level"].notna().any()

    if is_categorical:
        mask = (attr_rows["creative_attribute_level"].astype(str) == str(value)) & (
            attr_rows["p_value"] < threshold
        )
    else:
        mask = attr_rows["creative_attribute_level"].isna() & (attr_rows["p_value"] < threshold)

    significant = attr_rows[mask].copy()
    if significant.empty:
        return []

    significant = significant.sort_values("correlation", key=lambda s: s.abs(), ascending=False)

    # Group by (characteristic, sign) and join values with comma
    groups: dict[tuple[str, str], list[str]] = {}
    for _, r in significant.iterrows():
        sign = "pos" if r["correlation"] > 0 else "neg"
        key = (str(r["user_characteristic"]), sign)
        groups.setdefault(key, []).append(str(r["user_characteristic_value"]))

    tags = []
    for (char, sign), values in groups.items():
        emoji = "🟢" if sign == "pos" else "🔴"
        tags.append((emoji, f"{char}: {', '.join(values)}"))
    return tags


def _render_attributes_with_tags(
    creative_row: pd.Series,
    corr_df: pd.DataFrame,
) -> None:
    """Render the attributes grid with inline audience-signal tags."""
    st.subheader("Attributes & Audience Signals")
    st.caption(
        "🟢 = audience segment that correlates positively with this attribute  "
        "🔴 = audience segment that correlates negatively"
    )

    fields_left = _ATTRIBUTE_FIELDS[: len(_ATTRIBUTE_FIELDS) // 2 + 1]
    fields_right = _ATTRIBUTE_FIELDS[len(_ATTRIBUTE_FIELDS) // 2 + 1 :]

    attr_col1, attr_col2 = st.columns(2)

    for col, fields in ((attr_col1, fields_left), (attr_col2, fields_right)):
        with col:
            for col_key, label in fields:
                val = creative_row.get(col_key, None)
                if val is None or pd.isna(val):
                    continue
                tags = _explainability_tags(corr_df, col_key, val)
                tag_str = "  " + "  ".join(f"{e} {t}" for e, t in tags) if tags else ""
                st.markdown(f"**{label}:** {val}{tag_str}")


def _derive_target_segments(
    campaigns_df: pd.DataFrame,
    campaign_id: str,
    creative_row: pd.Series,
    advertiser: str,
) -> list[tuple[str, str]]:
    """Build target_segments from campaign metadata and the creative's vertical."""
    segments: list[tuple[str, str]] = []

    camp_rows = campaigns_df[
        (campaigns_df["campaign_id"] == campaign_id)
        & (campaigns_df["advertiser_name"] == advertiser)
    ]
    if not camp_rows.empty:
        camp = camp_rows.iloc[0]

        for field in ("target_age_segment", "objective"):
            val = camp.get(field)
            if pd.notna(val) and str(val).strip():
                segments.append((field, str(val).strip()))

        # "Both" means the campaign targets Android AND iOS — emit two segments
        # so the scorer can match correlations for each platform independently.
        target_os_val = camp.get("target_os")
        if pd.notna(target_os_val) and str(target_os_val).strip():
            os_str = str(target_os_val).strip()
            if os_str == "Both":
                segments.append(("target_os", "Android"))
                segments.append(("target_os", "iOS"))
            else:
                segments.append(("target_os", os_str))

        countries_raw = camp.get("countries", "")
        if pd.notna(countries_raw) and str(countries_raw).strip():
            for c in str(countries_raw).replace(",", "|").split("|"):
                c = c.strip()
                if c:
                    segments.append(("country", c))

    vertical = creative_row.get("vertical")
    if pd.notna(vertical) and str(vertical).strip():
        segments.append(("vertical", str(vertical).strip()))

    logger.debug(
        "_derive_target_segments: {} segments for campaign={}", len(segments), campaign_id
    )
    return segments


def _render_alternative_card(
    alt_row: pd.Series,
    scorer_name: str,
    scorer_desc: str,
    corr_df: pd.DataFrame,
    img_container: "st.delta_generator.DeltaGenerator",
    attr_container: "st.delta_generator.DeltaGenerator",
) -> None:
    """Render a single alternative creative card split across two columns.

    *img_container* receives the scorer label and image; *attr_container*
    receives the attribute list with audience-signal tags.
    """
    with img_container:
        st.markdown(f"**{scorer_name} scorer**")
        st.caption(scorer_desc)

        asset_file = alt_row.get("asset_file", "")
        img_path = _asset_path(asset_file) if asset_file else Path("/nonexistent")
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
        else:
            st.markdown(
                "<div style='height:120px;background:#eee;display:flex;"
                "align-items:center;justify-content:center;border-radius:8px'>"
                "No image</div>",
                unsafe_allow_html=True,
            )

    with attr_container:
        for col_key, label in _ATTRIBUTE_FIELDS:
            val = alt_row.get(col_key)
            if val is None or pd.isna(val):
                continue
            tags = _explainability_tags(corr_df, col_key, val)
            tag_str = "  " + "  ".join(f"{e} {t}" for e, t in tags) if tags else ""
            st.markdown(f"**{label}:** {val}{tag_str}")


def _campaign_params_for(campaigns_df: pd.DataFrame, campaign_id: str, advertiser: str) -> dict:
    """Return campaign-level parameters for the given campaign as a dict."""
    camp_rows = campaigns_df[
        (campaigns_df["campaign_id"] == campaign_id)
        & (campaigns_df["advertiser_name"] == advertiser)
    ]
    if camp_rows.empty:
        return {}
    camp = camp_rows.iloc[0]
    return {
        field: camp.get(field)
        for field in ("vertical", "objective", "target_age_segment", "target_os")
        if pd.notna(camp.get(field))
    }


def _render_alternatives(
    summary_df: pd.DataFrame,
    campaigns_df: pd.DataFrame,
    campaign_id: str,
    creative_id: str,
    current_row: pd.Series,
    corr_df: pd.DataFrame,
    advertiser: str,
) -> None:
    """Render the Alternatives section."""
    st.subheader("Alternative Recommendations")
    st.caption(
        "Each alternative is selected by a different fitness strategy to maximise "
        "audience alignment for this campaign's target segments."
    )

    target_segments = _derive_target_segments(campaigns_df, campaign_id, current_row, advertiser)

    if not target_segments:
        st.info("No target audience data available to compute alternatives.")
        logger.warning(
            "_render_alternatives: no target segments derived for campaign={}", campaign_id
        )
        return

    logger.info(
        "_render_alternatives: creative={} advertiser={} {} segments: {}",
        creative_id,
        advertiser,
        len(target_segments),
        target_segments,
    )

    # Campaign-level parameters from the current campaign — these should not
    # vary across alternatives since we're recommending creative changes only.
    current_camp_params = _campaign_params_for(campaigns_df, campaign_id, advertiser)
    if "vertical" not in current_camp_params:
        vertical = current_row.get("vertical")
        if pd.notna(vertical):
            current_camp_params["vertical"] = str(vertical)

    seen_ids: set[str] = {creative_id}
    alternatives: list[tuple[str, str, pd.Series]] = []

    for scorer_name, scorer, scorer_desc in _SCORERS:
        scores = _cached_fitness_scores(summary_df, corr_df, tuple(target_segments), scorer_name)
        candidates = scores[~scores.index.isin(seen_ids)]
        if candidates.empty:
            logger.warning("_render_alternatives: no candidates left for scorer={}", scorer_name)
            continue

        best_id = str(candidates.index[0])
        alt_rows = summary_df[summary_df["creative_id"] == best_id]
        if alt_rows.empty:
            logger.warning("_render_alternatives: best_id={} not found in summary_df", best_id)
            continue

        seen_ids.add(best_id)
        # Override campaign-level fields so the alternative is presented in the
        # context of the current campaign, not the alternative's origin campaign.
        alt_row = alt_rows.iloc[0].copy()
        for field, val in current_camp_params.items():
            if field in alt_row.index:
                alt_row[field] = val
        alternatives.append((scorer_name, scorer_desc, alt_row))

    if not alternatives:
        st.info("No alternatives could be computed for the current target audience.")
        return

    # Layout: [img1 | attrs1 | img2 | attrs2 | img3 | attrs3]
    cols = st.columns([1, 2] * len(alternatives))
    for i, (scorer_name, scorer_desc, alt_row) in enumerate(alternatives):
        img_col = cols[i * 2]
        attr_col = cols[i * 2 + 1]
        _render_alternative_card(alt_row, scorer_name, scorer_desc, corr_df, img_col, attr_col)


def render_ad_detail_view(
    summary_df: pd.DataFrame,
    advertiser: str,
    campaign_id: str,
    creative_id: str,
    campaigns_df: pd.DataFrame,
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
    campaigns_df:
        campaigns.csv data (ID-mapped), used to derive target audience segments.
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

    corr_df = load_correlations(method="statistical", metric="perf_score")

    left, right = st.columns([1, 3])

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

        _render_attributes_with_tags(row, corr_df)

    st.divider()
    _render_alternatives(
        summary_df, campaigns_df, campaign_id, creative_id, row, corr_df, advertiser
    )
