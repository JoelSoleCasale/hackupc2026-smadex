"""Comparison charts for the Smadex Creative Intelligence dashboard.

Four public functions, each returning a ``go.Figure``:
  - render_peer_rank_bar        — sector ranking bar chart
  - render_geo_heatmap          — choropleth by country
  - render_distribution_box     — box plot you vs sector
  - render_format_bar           — grouped bar by ad format
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from loguru import logger

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

COLOR_YOU = "#1e6fd9"
COLOR_PEERS = "#2a5a9f"

_ISO2_TO_ISO3 = {
    "US": "USA",
    "GB": "GBR",
    "DE": "DEU",
    "FR": "FRA",
    "ES": "ESP",
    "IT": "ITA",
    "NL": "NLD",
    "PL": "POL",
    "SE": "SWE",
    "NO": "NOR",
    "DK": "DNK",
    "FI": "FIN",
    "BE": "BEL",
    "AT": "AUT",
    "CH": "CHE",
    "PT": "PRT",
    "IE": "IRL",
    "CA": "CAN",
    "MX": "MEX",
    "BR": "BRA",
    "AR": "ARG",
    "CO": "COL",
    "CL": "CHL",
    "PE": "PER",
    "JP": "JPN",
    "KR": "KOR",
    "AU": "AUS",
    "NZ": "NZL",
    "SG": "SGP",
    "IN": "IND",
    "TH": "THA",
    "ID": "IDN",
    "PH": "PHL",
    "VN": "VNM",
    "MY": "MYS",
    "HK": "HKG",
    "TW": "TWN",
    "CN": "CHN",
    "ZA": "ZAF",
    "NG": "NGA",
    "EG": "EGY",
}

COUNTRY_LANGUAGE: dict[str, str] = {
    "US": "English",
    "GB": "English",
    "AU": "English",
    "CA": "English",
    "IE": "English",
    "NZ": "English",
    "ES": "Spanish",
    "MX": "Spanish",
    "AR": "Spanish",
    "CO": "Spanish",
    "CL": "Spanish",
    "PE": "Spanish",
    "DE": "German",
    "AT": "German",
    "CH": "German",
    "FR": "French",
    "BE": "French",
    "JP": "Japanese",
    "KR": "Korean",
    "BR": "Portuguese",
    "PT": "Portuguese",
    "IT": "Italian",
    "NL": "Dutch",
    "PL": "Polish",
    "SE": "Swedish",
    "NO": "Norwegian",
    "DK": "Danish",
    "FI": "Finnish",
    "IN": "Hindi",
    "TH": "Thai",
    "ID": "Indonesian",
    "SG": "English",
    "HK": "Chinese",
    "TW": "Chinese",
    "CN": "Chinese",
}

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _empty_fig(message: str = "No data") -> go.Figure:
    """Return a blank figure with a centred annotation."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 16, "color": "grey"},
    )
    fig.update_layout(xaxis_visible=False, yaxis_visible=False)
    return fig


def _resolve_metric(
    summary_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    metric: str,
) -> pd.Series:
    """Return a Series indexed by creative_id with the metric value per creative."""
    if metric == "perf_score":
        return summary_df.set_index("creative_id")["perf_score"]

    # Aggregate daily per creative
    agg = daily_df.groupby("creative_id").agg(
        impressions=("impressions", "sum"),
        clicks=("clicks", "sum"),
        conversions=("conversions", "sum"),
        revenue_usd=("revenue_usd", "sum"),
        spend_usd=("spend_usd", "sum"),
    )
    if metric == "ctr":
        return (agg["clicks"] / agg["impressions"].replace(0, float("nan"))).rename("ctr")
    elif metric == "cvr":
        return (agg["conversions"] / agg["clicks"].replace(0, float("nan"))).rename("cvr")
    elif metric == "roas":
        return (agg["revenue_usd"] / agg["spend_usd"].replace(0, float("nan"))).rename("roas")

    raise ValueError(f"Unknown metric: {metric!r}")


# ---------------------------------------------------------------------------
# Chart A — Sector ranking bar
# ---------------------------------------------------------------------------


def render_peer_rank_bar(
    your_summary: pd.DataFrame,
    your_daily: pd.DataFrame,
    filters: dict,
    sector_name: str,
    peers_summary: pd.DataFrame,
    peers_daily: pd.DataFrame,
) -> go.Figure:
    """Horizontal bar chart: median metric per advertiser, you vs 5 peers."""
    metric: str = filters["metric"]
    logger.debug("render_peer_rank_bar metric={}", metric)

    if your_summary.empty:
        return _empty_fig("No data for selected advertiser")

    combined_summary = pd.concat([your_summary, peers_summary], ignore_index=True)
    combined_daily = pd.concat([your_daily, peers_daily], ignore_index=True)

    if metric == "perf_score":
        median_series = combined_summary.groupby("advertiser_name")["perf_score"].median()
        adv_metric = median_series.reset_index()
        adv_metric.columns = ["advertiser_name", "metric_value"]
    else:
        metric_series = _resolve_metric(combined_summary, combined_daily, metric)
        # Join advertiser_name
        id_to_adv = combined_summary.set_index("creative_id")["advertiser_name"]
        merged = metric_series.to_frame("metric_value").join(id_to_adv)
        adv_metric = merged.groupby("advertiser_name")["metric_value"].median().reset_index()
        adv_metric.columns = ["advertiser_name", "metric_value"]

    adv_metric = adv_metric.sort_values("metric_value", ascending=False)

    # Identify your advertiser name(s)
    your_adv_names = set(your_summary["advertiser_name"].unique())

    fig = go.Figure()
    for i, row in adv_metric.iterrows():
        adv = row["advertiser_name"]
        val = row["metric_value"]
        fig.add_trace(
            go.Bar(
                x=[val],
                y=[adv],
                orientation="h",
                marker_color=COLOR_YOU if adv in your_adv_names else COLOR_PEERS,
                opacity=1.0 if adv in your_adv_names else 0.6,
                name=adv,
                showlegend=False,
            )
        )

    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        title=f"Sector ranking — {sector_name} — {metric}",
        xaxis_title=metric,
        yaxis_title="Advertiser",
        barmode="overlay",
        height=350,
    )
    return fig


# ---------------------------------------------------------------------------
# Chart B — Geo heatmap
# ---------------------------------------------------------------------------


def render_geo_heatmap(
    your_summary: pd.DataFrame,
    your_daily: pd.DataFrame,
    filters: dict,
    peers_summary: pd.DataFrame,
    peers_daily: pd.DataFrame,
) -> go.Figure:
    """Choropleth showing the selected metric by country for YOUR creatives."""
    metric: str = filters["metric"]
    logger.debug("render_geo_heatmap metric={}", metric)

    if your_daily.empty:
        return _empty_fig("No daily data available")

    if metric == "perf_score":
        if your_summary.empty:
            return _empty_fig("No summary data available")
        daily_with_perf = your_daily.merge(
            your_summary[["creative_id", "perf_score"]],
            on="creative_id",
            how="left",
        )
        country_metric = (
            daily_with_perf.groupby("country")
            .apply(
                lambda g: (
                    (g["perf_score"] * g["impressions"]).sum() / g["impressions"].sum()
                    if g["impressions"].sum() > 0
                    else float("nan")
                ),
                include_groups=False,
            )
            .rename("metric_value")
        )
    else:
        agg = your_daily.groupby("country").agg(
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
            conversions=("conversions", "sum"),
            revenue_usd=("revenue_usd", "sum"),
            spend_usd=("spend_usd", "sum"),
        )
        if metric == "ctr":
            country_metric = (agg["clicks"] / agg["impressions"].replace(0, float("nan"))).rename(
                "metric_value"
            )
        elif metric == "cvr":
            country_metric = (agg["conversions"] / agg["clicks"].replace(0, float("nan"))).rename(
                "metric_value"
            )
        elif metric == "roas":
            country_metric = (
                agg["revenue_usd"] / agg["spend_usd"].replace(0, float("nan"))
            ).rename("metric_value")
        else:
            return _empty_fig(f"Unknown metric: {metric}")

    country_metric = country_metric.dropna()

    if country_metric.empty:
        return _empty_fig("No country-level data")

    # Convert ISO-2 → ISO-3
    country_data = country_metric.reset_index()
    country_data.columns = ["country", "value"]
    country_data["iso3"] = country_data["country"].map(_ISO2_TO_ISO3)
    country_data = country_data.dropna(subset=["iso3"])

    if country_data.empty:
        return _empty_fig("No mappable country data")

    fig = px.choropleth(
        country_data,
        locations="iso3",
        locationmode="ISO-3",
        color="value",
        color_continuous_scale="Blues",
        title=f"{metric} by country",
    )
    fig.update_layout(height=350)
    return fig


# ---------------------------------------------------------------------------
# Chart C — Distribution violin
# ---------------------------------------------------------------------------


def render_distribution_box(
    your_summary: pd.DataFrame,
    your_daily: pd.DataFrame,
    filters: dict,
    peers_summary: pd.DataFrame,
    peers_daily: pd.DataFrame,
) -> go.Figure:
    """Box plot comparing metric distribution: your creatives vs sector."""
    metric: str = filters["metric"]
    logger.debug("render_distribution_box metric={}", metric)

    your_vals = (
        _resolve_metric(your_summary, your_daily, metric).dropna().tolist()
        if not your_summary.empty
        else []
    )
    peer_vals = (
        _resolve_metric(peers_summary, peers_daily, metric).dropna().tolist()
        if not peers_summary.empty
        else []
    )

    if not your_vals and not peer_vals:
        return _empty_fig("No data to display")

    fig = go.Figure()

    if peer_vals:
        fig.add_trace(
            go.Box(
                y=peer_vals,
                name="Sector creatives",
                boxpoints=False,
                fillcolor=COLOR_PEERS,
                line_color=COLOR_PEERS,
                opacity=0.6,
            )
        )

    if your_vals:
        fig.add_trace(
            go.Box(
                y=your_vals,
                name="Your creatives",
                boxpoints="all",
                fillcolor=COLOR_YOU,
                line_color=COLOR_YOU,
                opacity=0.8,
            )
        )

    fig.update_layout(
        title=f"{metric} distribution — you vs sector",
        yaxis_title=metric,
        boxmode="overlay",
        xaxis_title="",
        height=400,
    )
    return fig


# ---------------------------------------------------------------------------
# Chart D — Format grouped bar
# ---------------------------------------------------------------------------


def render_format_bar(
    your_summary: pd.DataFrame,
    your_daily: pd.DataFrame,
    filters: dict,
    peers_summary: pd.DataFrame,
    peers_daily: pd.DataFrame,
) -> go.Figure:
    """Grouped bar chart: metric by ad format, you vs sector peer average."""
    metric: str = filters["metric"]
    logger.debug("render_format_bar metric={}", metric)

    def _format_medians(summary_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.Series:
        """Per-format median of the metric."""
        if summary_df.empty:
            return pd.Series(dtype=float)
        metric_series = _resolve_metric(summary_df, daily_df, metric)
        fmt_map = summary_df.set_index("creative_id")["format"]
        merged = metric_series.to_frame("val").join(fmt_map)
        return merged.groupby("format")["val"].median()

    your_fmt = _format_medians(your_summary, your_daily)
    peer_fmt = _format_medians(peers_summary, peers_daily)

    all_formats = sorted(set(your_fmt.index) | set(peer_fmt.index))

    if not all_formats:
        return _empty_fig("No format data available")

    rows = []
    for fmt in all_formats:
        if fmt in your_fmt.index:
            rows.append({"format": fmt, "value": your_fmt[fmt], "group": "You"})
        if fmt in peer_fmt.index:
            rows.append({"format": fmt, "value": peer_fmt[fmt], "group": "Sector avg"})

    df_plot = pd.DataFrame(rows)

    color_map = {"You": COLOR_YOU, "Sector avg": COLOR_PEERS}

    fig = px.bar(
        df_plot,
        x="format",
        y="value",
        color="group",
        barmode="group",
        color_discrete_map=color_map,
        title=f"{metric} by format",
        labels={"value": metric, "format": "Ad Format", "group": ""},
    )
    fig.update_layout(height=380)
    return fig
