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
from src.models.predict import predict_fatigue_day, predict_profitability_end

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

_PURPLE = "#881CA6"
_PURPLE_LIGHT = "#f5e9f9"
_PURPLE_DARK = "#6a1580"

_STATUS_STYLES: dict[str, tuple[str, str, str]] = {
    "top_performer": ("🟢", "#16a34a", "#f0fdf4"),
    "stable": ("🔵", "#2563eb", "#eff6ff"),
    "underperformer": ("🟡", "#d97706", "#fffbeb"),
    "fatigued": ("🔴", "#dc2626", "#fef2f2"),
}

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
    ("Linear", LinearFitnessScorer(), "maximises total alignment across all audience signals"),
    (
        "Sharpe",
        SharpeCorrelationScorer(),
        "rewards creatives with consistent alignment across attributes",
    ),
    ("Top-10", TopKFitnessScorer(k=10), "focuses on the 10 strongest audience signals"),
]

_BOOL_FIELDS = {"has_gameplay", "has_ugc_style", "has_price", "has_discount_badge"}
_CHAR_DISPLAY = {"target_age_segment": "ages", "target_os": "os"}
_AGE_ORDER = ["18-24", "25-34", "35-44", "45-54"]


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _asset_path(asset_file: str) -> Path:
    return _PROJECT_ROOT / "data" / asset_file


def _merge_age_ranges(ages: list[str]) -> str:
    present = [a for a in _AGE_ORDER if a in ages]
    if not present:
        return ", ".join(ages)
    indices = [_AGE_ORDER.index(a) for a in present]
    if indices == list(range(min(indices), max(indices) + 1)):
        return f"{present[0].split('-')[0]}-{present[-1].split('-')[1]}"
    return ", ".join(present)


def _format_val(col_key: str, val: object) -> str:
    if col_key in _BOOL_FIELDS:
        return "Yes" if float(val) == 1 else "No"
    return str(val)


def _metric_box(label: str, value: str, *, purple: bool = False) -> str:
    if purple:
        bg, border, label_color, val_color = "#f5e9f9", "#d4a8e6", _PURPLE, _PURPLE
    else:
        bg, border, label_color, val_color = "#f3f4f6", "#d1d5db", "#6b7280", "#374151"
    return (
        f'<div style="background:{bg};border:1px solid {border};border-radius:10px;'
        f'padding:10px 14px;text-align:center;margin-bottom:4px">'
        f'<div style="font-size:11px;color:{label_color};font-weight:600;margin-bottom:3px">{label}</div>'
        f'<div style="font-size:20px;font-weight:700;color:{val_color}">{value}</div>'
        f"</div>"
    )


def _section_header(title: str) -> None:
    st.markdown(
        f'<div style="border-left:4px solid {_PURPLE};padding-left:10px;margin:12px 0 6px">'
        f'<span style="font-size:17px;font-weight:700">{title}</span></div>',
        unsafe_allow_html=True,
    )


def _status_badge(status: str) -> None:
    icon, text_color, bg_color = _STATUS_STYLES.get(status, ("⚪", "#6b7280", "#f9fafb"))
    label = status.replace("_", " ").title()
    st.markdown(
        f'<div style="display:inline-flex;align-items:center;gap:6px;'
        f"background:{bg_color};border:1px solid {text_color}40;"
        f"color:{text_color};padding:5px 14px;border-radius:20px;"
        f'font-weight:600;font-size:13px;margin-top:8px">'
        f"{icon} {label}</div>",
        unsafe_allow_html=True,
    )


def _render_tags_html(tags: list[tuple[str, str]]) -> str:
    parts = []
    for sign, text in tags:
        color = "#16a34a" if sign == "pos" else "#dc2626"
        bg = "#f0fdf4" if sign == "pos" else "#fef2f2"
        indicator = "+" if sign == "pos" else "−"
        parts.append(
            f'<span style="background:{bg};color:{color};border:1px solid {color}40;'
            f'border-radius:10px;padding:1px 7px;font-size:11px;font-weight:600;white-space:nowrap">'
            f"{indicator} {text}</span>"
        )
    return "&nbsp;".join(parts)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def _daily_roas_agg(daily_df: pd.DataFrame, creative_id: str) -> pd.DataFrame:
    agg = (
        daily_df[daily_df["creative_id"] == creative_id]
        .groupby("days_since_launch", as_index=False)
        .agg(revenue=("revenue_usd", "sum"), spend=("spend_usd", "sum"))
    )
    agg["daily_roas"] = agg["revenue"] / agg["spend"].replace(0, float("nan"))
    return agg.sort_values("days_since_launch")


def _roas_chart(agg: pd.DataFrame, predicted_profitability_end: float | None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=agg["days_since_launch"],
            y=agg["daily_roas"],
            mode="lines+markers",
            name="Daily ROAS",
            line=dict(color=_PURPLE, width=2),
            marker=dict(size=5, color=_PURPLE),
            fill="tozeroy",
            fillcolor="rgba(136,28,166,0.07)",
        )
    )
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="#dc2626",
        annotation_text="Break-even (ROAS = 1)",
        annotation_position="bottom right",
        annotation_font_color="#dc2626",
    )
    if predicted_profitability_end is not None and not math.isnan(predicted_profitability_end):
        fig.add_vline(
            x=predicted_profitability_end,
            line_dash="dot",
            line_color="#d97706",
            annotation_text=f"End profitability: day {int(predicted_profitability_end)}",
            annotation_position="top left",
            annotation_font_color="#d97706",
        )
    fig.update_layout(
        title=dict(
            text="ROAS over time  <sup><i style='font-size:11px;color:#888'>"
            "⏱ vertical marker predicted from first 7 days</i></sup>",
            font=dict(size=14),
        ),
        xaxis_title="Days Since Launch",
        yaxis_title="ROAS",
        height=280,
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(gridcolor="#f0e6f9", showgrid=True)
    fig.update_yaxes(gridcolor="#f0e6f9", showgrid=True)
    return fig


_OS_COLORS = [_PURPLE, "#e07b39", "#2ca02c", "#d62728", "#9467bd"]


def _render_creative_roas_evolution(daily_df: pd.DataFrame, creative_id: str) -> go.Figure:
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
        fig.update_layout(xaxis_visible=False, yaxis_visible=False, height=300)
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
            line={"color": _PURPLE, "width": 2},
            marker={"size": 4, "color": _PURPLE},
            fill="tozeroy",
            fillcolor="rgba(136,28,166,0.07)",
            hovertemplate="%{x|%b %d}<br>ROAS: %{y:.2f}×<extra></extra>",
        )
    )
    fig.update_layout(
        title="ROAS Evolution",
        xaxis_title="Date",
        yaxis_title="ROAS",
        height=300,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(gridcolor="#f0e6f9")
    fig.update_yaxes(gridcolor="#f0e6f9")
    return fig


def _render_os_breakdown(daily_df: pd.DataFrame, creative_id: str) -> go.Figure:
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
        fig.update_layout(xaxis_visible=False, yaxis_visible=False, height=300)
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

    for i, os_name in enumerate(sorted(agg["os"].unique())):
        os_row = agg[agg["os"] == os_name].iloc[0]
        values = [os_row["ctr"], os_row["cvr"]]
        fig.add_trace(
            go.Bar(
                x=["CTR (%)", "CVR (%)"],
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
        height=300,
        yaxis={"range": [0, agg[["ctr", "cvr"]].max().max() * 1.25]},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_yaxes(gridcolor="#f0e6f9")
    return fig


# ---------------------------------------------------------------------------
# Attributes
# ---------------------------------------------------------------------------


def _explainability_tags(
    corr_df: pd.DataFrame,
    attribute: str,
    value: object,
    threshold: float = 0.01,
) -> list[tuple[str, str]]:
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

    if not is_categorical and float(value) == 0:
        significant["correlation"] = -significant["correlation"]

    significant = significant.sort_values("correlation", key=lambda s: s.abs(), ascending=False)

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


def _render_attributes_with_tags(creative_row: pd.Series, corr_df: pd.DataFrame) -> None:
    _section_header("Attributes & Audience Signals")
    st.caption(
        '<span style="color:#888">🟢 + audience correlates positively &nbsp;&nbsp;'
        "🔴 − correlates negatively</span>",
        unsafe_allow_html=True,
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
                display_val = _format_val(col_key, val)
                tags = _explainability_tags(corr_df, col_key, val)
                tag_html = ("&nbsp;&nbsp;" + _render_tags_html(tags)) if tags else ""
                st.markdown(f"**{label}:** {display_val}{tag_html}", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Alternatives
# ---------------------------------------------------------------------------


def _derive_target_segments(
    campaigns_df: pd.DataFrame,
    campaign_id: str,
    creative_row: pd.Series,
    advertiser: str,
) -> list[tuple[str, str]]:
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
        target_os_val = camp.get("target_os")
        if pd.notna(target_os_val) and str(target_os_val).strip():
            os_str = str(target_os_val).strip()
            if os_str == "Both":
                segments.extend([("target_os", "Android"), ("target_os", "iOS")])
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
    return segments


def _campaign_params_for(campaigns_df: pd.DataFrame, campaign_id: str, advertiser: str) -> dict:
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


_SCORER_ICONS = {"Linear": "⚖️", "Sharpe": "📐", "Top-10": "🏆"}


def _render_alternative_card(
    alt_row: pd.Series,
    scorer_name: str,
    scorer_desc: str,
    corr_df: pd.DataFrame,
    index: int,
) -> None:
    """Render one alternative as a full-width bordered card."""
    icon = _SCORER_ICONS.get(scorer_name, "✦")
    with st.container(border=True):
        # Coloured header strip
        st.markdown(
            f'<div style="background:linear-gradient(90deg,{_PURPLE},{_PURPLE_DARK});'
            f"padding:10px 16px;border-radius:6px;margin-bottom:12px;display:flex;"
            f'align-items:center;gap:10px">'
            f'<span style="font-size:20px">{icon}</span>'
            f'<div><span style="color:#fff;font-weight:700;font-size:15px">'
            f"Option {index} — {scorer_name} Strategy</span><br>"
            f'<span style="color:rgba(255,255,255,.75);font-size:12px">{scorer_desc}</span>'
            f"</div></div>",
            unsafe_allow_html=True,
        )

        asset_file = alt_row.get("asset_file", "")
        img_path = _asset_path(asset_file) if asset_file else Path("/nonexistent")
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
        else:
            st.markdown(
                f'<div style="height:160px;background:{_PURPLE_LIGHT};display:flex;'
                f"align-items:center;justify-content:center;border-radius:8px;"
                f'color:{_PURPLE};font-size:13px">No image available</div>',
                unsafe_allow_html=True,
            )

        # Key metrics pill row
        perf = alt_row.get("perf_score")
        roas = alt_row.get("overall_roas")
        if pd.notna(perf) or pd.notna(roas):
            pills = []
            if pd.notna(perf):
                pills.append(
                    f'<span style="background:{_PURPLE_LIGHT};color:{_PURPLE};'
                    f'border-radius:10px;padding:2px 9px;font-size:12px;font-weight:600">'
                    f"Perf {float(perf):.2f}</span>"
                )
            if pd.notna(roas):
                pills.append(
                    f'<span style="background:#f0fdf4;color:#16a34a;'
                    f'border-radius:10px;padding:2px 9px;font-size:12px;font-weight:600">'
                    f"ROAS {float(roas):.2f}×</span>"
                )
            st.markdown(
                '<div style="display:flex;flex-wrap:wrap;gap:5px;margin-top:8px;margin-bottom:10px">'
                + "".join(pills)
                + "</div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<p style="font-weight:600;color:{_PURPLE};margin-bottom:6px">Creative Attributes</p>',
            unsafe_allow_html=True,
        )
        for col_key, label in _ATTRIBUTE_FIELDS:
            val = alt_row.get(col_key)
            if val is None or pd.isna(val):
                continue
            display_val = _format_val(col_key, val)
            tags = _explainability_tags(corr_df, col_key, val)
            tag_html = ("&nbsp;&nbsp;" + _render_tags_html(tags)) if tags else ""
            st.markdown(f"**{label}:** {display_val}{tag_html}", unsafe_allow_html=True)


def _render_alternatives(
    summary_df: pd.DataFrame,
    campaigns_df: pd.DataFrame,
    campaign_id: str,
    creative_id: str,
    current_row: pd.Series,
    corr_df: pd.DataFrame,
    advertiser: str,
) -> None:
    _section_header("Alternative Recommendations")
    st.caption(
        "Each alternative is selected by a different fitness strategy to maximise "
        "audience alignment for this campaign's target segments."
    )

    target_segments = _derive_target_segments(campaigns_df, campaign_id, current_row, advertiser)
    if not target_segments:
        st.info("No target audience data available to compute alternatives.")
        return

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
            continue
        best_id = str(candidates.index[0])
        alt_rows = summary_df[summary_df["creative_id"] == best_id]
        if alt_rows.empty:
            continue
        seen_ids.add(best_id)
        alt_row = alt_rows.iloc[0].copy()
        for field, val in current_camp_params.items():
            if field in alt_row.index:
                alt_row[field] = val
        alternatives.append((scorer_name, scorer_desc, alt_row))

    if not alternatives:
        st.info("No alternatives could be computed for the current target audience.")
        return

    cols = st.columns(len(alternatives))
    for i, ((scorer_name, scorer_desc, alt_row), col) in enumerate(zip(alternatives, cols), 1):
        with col:
            _render_alternative_card(alt_row, scorer_name, scorer_desc, corr_df, i)


# ---------------------------------------------------------------------------
# Main view
# ---------------------------------------------------------------------------


def render_ad_detail_view(
    summary_df: pd.DataFrame,
    advertiser: str,
    campaign_id: str,
    creative_id: str,
    campaigns_df: pd.DataFrame,
    daily_df: pd.DataFrame | None = None,
) -> None:
    """Render the ad detail page."""
    if st.button("← Back to Campaign"):
        st.session_state.current_view = "campaign"
        st.session_state.selected_creative = None
        st.rerun()

    st.markdown(
        f'<p style="color:#888;font-size:13px;margin-bottom:0">'
        f"<b>{advertiser}</b> › {campaign_id} › {creative_id}</p>",
        unsafe_allow_html=True,
    )
    st.title(creative_id)

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

    # --- Predictions ---
    days_running: int = 0
    daily_agg: pd.DataFrame = pd.DataFrame()
    pred_profitability: float | None = None
    pred_fatigue: float | None = None
    mae_profitability: float = float("nan")
    mae_fatigue: float = float("nan")

    if daily_df is not None and not daily_df.empty:
        adv_daily = (
            daily_df[daily_df["advertiser_name"] == advertiser]
            if "advertiser_name" in daily_df.columns
            else daily_df
        )
        daily_agg = _daily_roas_agg(adv_daily, creative_id)
        if not daily_agg.empty:
            days_running = int(daily_agg["days_since_launch"].max())

        def _scalar(series: pd.Series, key: str) -> float | None:
            raw = series.get(key)
            if raw is None:
                return None
            val = float(raw.iloc[0]) if isinstance(raw, pd.Series) else float(raw)
            return val if not math.isnan(val) else None

        prof_series, mae_profitability = predict_profitability_end(daily_df, summary_df)
        fat_series, mae_fatigue = predict_fatigue_day(daily_df, summary_df)
        pred_profitability = _scalar(prof_series, creative_id)
        pred_fatigue = _scalar(fat_series, creative_id)

    # --- Layout: image left | details right ---
    left, right = st.columns([1, 3])

    with left:
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
        else:
            st.markdown(
                f'<div style="height:300px;background:{_PURPLE_LIGHT};display:flex;'
                f"align-items:center;justify-content:center;border-radius:8px;"
                f'color:{_PURPLE};font-size:13px">No image available</div>',
                unsafe_allow_html=True,
            )
        _status_badge(status)

    with right:
        _section_header("General Info")
        perf = row.get("perf_score")
        ctr = row.get("overall_ctr")
        cvr = row.get("overall_cvr")
        roas = row.get("overall_roas")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                _metric_box(
                    "Perf Score", f"{float(perf):.3f}" if pd.notna(perf) else "N/A", purple=True
                ),
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                _metric_box("CTR", f"{float(ctr) * 100:.2f}%" if pd.notna(ctr) else "N/A"),
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                _metric_box("CVR", f"{float(cvr) * 100:.2f}%" if pd.notna(cvr) else "N/A"),
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                _metric_box("ROAS", f"{float(roas):.2f}×" if pd.notna(roas) else "N/A"),
                unsafe_allow_html=True,
            )

        # Predictions — only after 7 days
        if days_running >= 7:
            fa1, fa2, fa3 = st.columns(3)
            with fa1:
                if pred_profitability is not None:
                    st.metric(
                        "Est. Profitability End",
                        f"day {pred_profitability:.0f}",
                        delta=f"±{round(mae_profitability)} days",
                        delta_color="off",
                    )
            with fa2:
                if pred_fatigue is not None:
                    st.metric(
                        "Est. Fatigue Day",
                        f"day {pred_fatigue:.0f}",
                        delta=f"±{round(mae_fatigue)} days",
                        delta_color="off",
                    )
            with fa3:
                st.metric("Days Active", f"{days_running} days")

            st.caption("⏱ Predictions use only the first 7 days of creative data.")

            if not daily_agg.empty:
                st.plotly_chart(
                    _roas_chart(daily_agg, pred_profitability), use_container_width=True
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
