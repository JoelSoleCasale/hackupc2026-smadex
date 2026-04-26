"""Ad detail view for the Smadex Creative Intelligence dashboard."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from loguru import logger

from src.analysis.fitness_scorer import (
    LinearFitnessScorer,
    SharpeCorrelationScorer,
    TopKFitnessScorer,
)
from src.data.loader import load_correlations
from src.models.predict import MAE_DAYS, predict_fatigue_days

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

_BOOL_FIELDS = {"has_gameplay", "has_ugc_style", "has_price", "has_discount_badge"}

_CHAR_DISPLAY = {
    "target_age_segment": "ages",
    "target_os": "os",
}

_AGE_ORDER = ["18-24", "25-34", "35-44", "45-54"]


def _merge_age_ranges(ages: list[str]) -> str:
    """Collapse consecutive age brackets into a single range, e.g. ['18-24','25-34'] → '18-54'."""
    present = [a for a in _AGE_ORDER if a in ages]
    if not present:
        return ", ".join(ages)
    indices = [_AGE_ORDER.index(a) for a in present]
    if indices == list(range(min(indices), max(indices) + 1)):
        return f"{present[0].split('-')[0]}-{present[-1].split('-')[1]}"
    return ", ".join(present)


def _format_val(col_key: str, val: object) -> str:
    """Human-friendly value string for a creative attribute."""
    if col_key in _BOOL_FIELDS:
        return "Yes" if float(val) == 1 else "No"
    return str(val)


def _render_tags_html(tags: list[tuple[str, str]]) -> str:
    """Return an HTML string of compact +/− signal chips."""
    parts = []
    for sign, text in tags:
        color = "#2ca02c" if sign == "pos" else "#d62728"
        indicator = "+" if sign == "pos" else "−"
        parts.append(
            f'<span style="color:{color};font-weight:700;font-size:12px">{indicator}</span>'
            f'<span style="color:#aaa;font-size:11px"> {text}</span>'
        )
    return "&nbsp;&nbsp;".join(parts)


def _asset_path(asset_file: str) -> Path:
    return _PROJECT_ROOT / "data" / asset_file


def _daily_roas_agg(daily_df: pd.DataFrame, creative_id: str) -> pd.DataFrame:
    """Aggregate daily ROAS across country/OS for a single creative."""
    agg = (
        daily_df[daily_df["creative_id"] == creative_id]
        .groupby("days_since_launch", as_index=False)
        .agg(revenue=("revenue_usd", "sum"), spend=("spend_usd", "sum"))
    )
    agg["daily_roas"] = agg["revenue"] / agg["spend"].replace(0, float("nan"))
    return agg.sort_values("days_since_launch")


def _roas_chart(agg: pd.DataFrame, predicted_fatigue_day: float | None) -> go.Figure:
    """ROAS-over-time line chart with break-even line and optional fatigue marker."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=agg["days_since_launch"],
            y=agg["daily_roas"],
            mode="lines+markers",
            name="Daily ROAS",
            line=dict(color="#4C78A8", width=2),
            marker=dict(size=5),
        )
    )
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="red",
        annotation_text="Break-even (ROAS = 1)",
        annotation_position="bottom right",
    )
    if predicted_fatigue_day is not None and not math.isnan(predicted_fatigue_day):
        fig.add_vline(
            x=predicted_fatigue_day,
            line_dash="dot",
            line_color="orange",
            annotation_text=f"Predicted fatigue: day {int(predicted_fatigue_day)}",
            annotation_position="top left",
        )
    fig.update_layout(
        xaxis_title="Days Since Launch",
        yaxis_title="ROAS",
        height=260,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


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

    # For binary attributes, flip the correlation sign when the creative lacks the feature.
    # corr(has_X, perf) > 0 means *having* X helps; absence means the opposite signal.
    if not is_categorical and float(value) == 0:
        significant["correlation"] = -significant["correlation"]

    significant = significant.sort_values("correlation", key=lambda s: s.abs(), ascending=False)

    # Group by (characteristic, sign) and join values with comma
    groups: dict[tuple[str, str], list[str]] = {}
    for _, r in significant.iterrows():
        sign = "pos" if r["correlation"] > 0 else "neg"
        key = (str(r["user_characteristic"]), sign)
        groups.setdefault(key, []).append(str(r["user_characteristic_value"]))

    tags = []
    for (char, sign), values in groups.items():
        display_char = _CHAR_DISPLAY.get(char, char)
        joined = _merge_age_ranges(values) if char == "target_age_segment" else ", ".join(values)
        tags.append((sign, f"{display_char}: {joined}"))
    return tags


def _render_attributes_with_tags(
    creative_row: pd.Series,
    corr_df: pd.DataFrame,
) -> None:
    """Render the attributes grid with inline audience-signal tags."""
    st.subheader("Attributes & Audience Signals")
    st.caption("+ audience segment correlates positively · − correlates negatively")

    fields_left = _ATTRIBUTE_FIELDS[: len(_ATTRIBUTE_FIELDS) // 2 + 1]
    fields_right = _ATTRIBUTE_FIELDS[len(_ATTRIBUTE_FIELDS) // 2 + 1 :]

    attr_col1, attr_col2 = st.columns(2)

    for col, fields in ((attr_col1, fields_left), (attr_col2, fields_right)):
        with col:
            for col_key, label in fields:
                val = creative_row.get(col_key, None)
                if val is None or pd.isna(val):
                    continue
                display_val = _format_val(col_key, val)
                tags = _explainability_tags(corr_df, col_key, val)
                tag_html = ("&nbsp;&nbsp;" + _render_tags_html(tags)) if tags else ""
                st.markdown(f"**{label}:** {display_val}{tag_html}", unsafe_allow_html=True)


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
            display_val = _format_val(col_key, val)
            tags = _explainability_tags(corr_df, col_key, val)
            tag_html = ("&nbsp;&nbsp;" + _render_tags_html(tags)) if tags else ""
            st.markdown(f"**{label}:** {display_val}{tag_html}", unsafe_allow_html=True)


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


_OS_COLORS = ["#1e6fd9", "#e07b39", "#2ca02c", "#d62728", "#9467bd"]


def _render_creative_roas_evolution(daily_df: pd.DataFrame, creative_id: str) -> go.Figure:
    """Single-line ROAS over time for one creative."""
    cdata = daily_df[daily_df["creative_id"] == creative_id].copy()
    fig = go.Figure()
    if cdata.empty:
        fig.add_annotation(
            text="No daily data",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 14, "color": "grey"},
        )
        fig.update_layout(xaxis_visible=False, yaxis_visible=False, height=380)
        return fig

    cdata["date"] = pd.to_datetime(cdata["date"])
    agg = (
        cdata.groupby("date")
        .agg(revenue=("revenue_usd", "sum"), spend=("spend_usd", "sum"))
        .reset_index()
    )
    agg["roas"] = agg["revenue"] / agg["spend"].replace(0, float("nan"))
    agg = agg.sort_values("date")

    fig.add_trace(
        go.Scatter(
            x=agg["date"],
            y=agg["roas"],
            mode="lines+markers",
            line={"color": "#1e6fd9", "width": 2},
            marker={"size": 4},
            fill="tozeroy",
            fillcolor="rgba(30,111,217,0.08)",
            hovertemplate="%{x|%b %d}<br>ROAS: %{y:.2f}×<extra></extra>",
        )
    )
    fig.update_layout(
        title="ROAS Evolution",
        xaxis_title="Date",
        yaxis_title="ROAS",
        height=380,
    )
    return fig


def _render_os_breakdown(daily_df: pd.DataFrame, creative_id: str) -> go.Figure:
    """Grouped bar: CTR and CVR on x-axis, one bar series per OS with value labels."""
    cdata = daily_df[daily_df["creative_id"] == creative_id].copy()
    fig = go.Figure()
    if cdata.empty:
        fig.add_annotation(
            text="No daily data",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 14, "color": "grey"},
        )
        fig.update_layout(xaxis_visible=False, yaxis_visible=False, height=380)
        return fig

    agg = (
        cdata.groupby("os")
        .agg(
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
            conversions=("conversions", "sum"),
        )
        .reset_index()
    )
    agg["ctr"] = agg["clicks"] / agg["impressions"].replace(0, float("nan")) * 100
    agg["cvr"] = agg["conversions"] / agg["clicks"].replace(0, float("nan")) * 100

    metrics = ["CTR (%)", "CVR (%)"]
    metric_keys = ["ctr", "cvr"]

    for i, os_name in enumerate(sorted(agg["os"].unique())):
        os_row = agg[agg["os"] == os_name].iloc[0]
        values = [os_row[k] for k in metric_keys]
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=values,
                name=os_name,
                marker_color=_OS_COLORS[i % len(_OS_COLORS)],
                text=[f"{v:.2f}%" for v in values],
                textposition="outside",
                hovertemplate=f"{os_name}<br>%{{x}}: %{{y:.2f}}%<extra></extra>",
            )
        )

    fig.update_layout(
        title="Performance by OS",
        xaxis_title="Metric",
        yaxis_title="Rate (%)",
        barmode="group",
        height=380,
        yaxis={"range": [0, agg[metric_keys].max().max() * 1.25]},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig


def render_ad_detail_view(
    summary_df: pd.DataFrame,
    advertiser: str,
    campaign_id: str,
    creative_id: str,
    campaigns_df: pd.DataFrame,
    daily_df: pd.DataFrame | None = None,
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
    daily_df:
        Full daily stats (all advertisers, ID-mapped), used for time-series plots and fatigue prediction.
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

    # --- Fatigue prediction (computed once for all creatives, cached) ---
    pred_fatigue_day: float | None = None
    days_running: int = 0
    daily_agg: pd.DataFrame = pd.DataFrame()

    if daily_df is not None and not daily_df.empty:
        daily_agg = _daily_roas_agg(daily_df, creative_id)
        if not daily_agg.empty:
            days_running = int(daily_agg["days_since_launch"].max())

        all_preds = predict_fatigue_days(daily_df, summary_df)
        raw = all_preds.get(creative_id)
        if raw is not None:
            # .get() returns a Series when index has duplicates (same mapped ID across advertisers)
            scalar = float(raw.iloc[0]) if isinstance(raw, pd.Series) else float(raw)
            if not math.isnan(scalar):
                pred_fatigue_day = scalar

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

        # Only show fatigue prediction and ROAS chart after 7 days of data exist
        if days_running >= 7:
            fa1, fa2 = st.columns(2)
            with fa1:
                if pred_fatigue_day is not None:
                    st.metric(
                        "Est. Profitability End",
                        f"day {pred_fatigue_day:.0f}",
                        delta=f"±{round(MAE_DAYS)} days",
                        delta_color="off",
                    )
            with fa2:
                st.metric("Days Active", f"{days_running} days")

            if not daily_agg.empty:
                st.plotly_chart(
                    _roas_chart(daily_agg, pred_fatigue_day),
                    use_container_width=True,
                )

        st.divider()

        _render_attributes_with_tags(row, corr_df)

    if daily_df is not None and not daily_df.empty:
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                _render_creative_roas_evolution(daily_df, creative_id), use_container_width=True
            )
        with col2:
            st.plotly_chart(_render_os_breakdown(daily_df, creative_id), use_container_width=True)

    st.divider()
    _render_alternatives(
        summary_df, campaigns_df, campaign_id, creative_id, row, corr_df, advertiser
    )
