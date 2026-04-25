"""
Build a single, self-contained HTML report covering all EDA + Gemma 4 embedding analysis.

Usage:
    # With embeddings (full report):
    python eda/build_comprehensive_report.py

    # Without embeddings (EDA-only sections):
    python eda/build_comprehensive_report.py --skip-embeddings

Output:
    eda/comprehensive_report.html
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
EDA_DIR = Path(__file__).resolve().parent
EMB_PATH = ROOT / "embeddings" / "creative_embeddings.npz"

PALETTE = px.colors.qualitative.Set2
STATUS_COLORS = {
    "top_performer": "#2ecc71",
    "stable": "#3498db",
    "fatigued": "#e67e22",
    "underperformer": "#e74c3c",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(ROOT / "creative_summary.csv")
    campaigns = pd.read_csv(ROOT / "campaigns.csv")
    advertisers = pd.read_csv(ROOT / "advertisers.csv")
    return df, campaigns, advertisers


def load_daily() -> pd.DataFrame:
    daily = pd.read_csv(ROOT / "creative_daily_country_os_stats.csv")
    return daily


def load_embeddings() -> tuple[np.ndarray, np.ndarray] | None:
    if not EMB_PATH.exists():
        return None
    data = np.load(EMB_PATH)
    return data["embeddings"], data["creative_ids"]


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------


def fig_to_html(fig: go.Figure, div_id: str) -> str:
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id=div_id,
        config={"displayModeBar": True, "responsive": True},
    )


# ---------------------------------------------------------------------------
# Section 1 – Executive Summary
# ---------------------------------------------------------------------------


def section_executive_summary(df: pd.DataFrame) -> str:
    status_counts = df["creative_status"].value_counts()
    total = len(df)

    kpis = {
        "Total Creatives": f"{total:,}",
        "Top Performers": f"{status_counts.get('top_performer', 0):,} ({status_counts.get('top_performer', 0) / total * 100:.0f}%)",
        "Avg CTR": f"{df['overall_ctr'].mean():.3%}",
        "Avg CVR": f"{df['overall_cvr'].mean():.3%}",
        "Avg ROAS": f"{df['overall_roas'].mean():.2f}x",
        "Total Impressions": f"{df['total_impressions'].sum() / 1e6:.1f}M",
        "Total Revenue": f"${df['total_revenue_usd'].sum() / 1e6:.1f}M",
        "Verticals": str(df["vertical"].nunique()),
    }

    fig_pie = px.pie(
        values=status_counts.values,
        names=status_counts.index,
        title="Creative Status Distribution",
        color=status_counts.index,
        color_discrete_map=STATUS_COLORS,
        hole=0.4,
    )
    fig_pie.update_layout(height=320, margin=dict(t=40, b=10, l=10, r=10))

    fig_roas = px.box(
        df,
        x="creative_status",
        y="overall_roas",
        color="creative_status",
        color_discrete_map=STATUS_COLORS,
        title="ROAS Distribution by Status",
        category_orders={
            "creative_status": ["top_performer", "stable", "fatigued", "underperformer"]
        },
    )
    fig_roas.update_layout(height=320, showlegend=False, margin=dict(t=40, b=10, l=10, r=10))

    kpi_cards = "".join(
        f'<div class="kpi-card"><div class="kpi-value">{v}</div>'
        f'<div class="kpi-label">{k}</div></div>'
        for k, v in kpis.items()
    )

    return f"""
<section id="executive-summary">
  <h2>1. Executive Summary</h2>
  <div class="kpi-grid">{kpi_cards}</div>
  <div class="chart-row">
    <div class="chart-half">{fig_to_html(fig_pie, "pie-status")}</div>
    <div class="chart-half">{fig_to_html(fig_roas, "box-roas")}</div>
  </div>
</section>"""


# ---------------------------------------------------------------------------
# Section 2 – Dataset Overview
# ---------------------------------------------------------------------------


def section_dataset_overview(df: pd.DataFrame) -> str:
    fig_vertical = px.histogram(
        df,
        x="vertical",
        color="creative_status",
        barmode="stack",
        color_discrete_map=STATUS_COLORS,
        title="Creatives per Vertical (by Status)",
        category_orders={
            "creative_status": ["top_performer", "stable", "fatigued", "underperformer"]
        },
    )
    fig_vertical.update_layout(height=340, margin=dict(t=40, b=10, l=10, r=10))

    fig_format = px.histogram(
        df,
        x="format",
        color="creative_status",
        barmode="group",
        color_discrete_map=STATUS_COLORS,
        title="Creatives per Format (by Status)",
    )
    fig_format.update_layout(height=340, margin=dict(t=40, b=10, l=10, r=10))

    table_data = {
        "Table": ["creative_summary", "campaigns", "advertisers", "daily_stats", "assets"],
        "Rows": ["1,080", "180", "36", "192,315", "1,080 PNGs"],
        "Key Columns": [
            "creative_id, perf_score, creative_status, design scores",
            "campaign_id, vertical, budget, target_os, target_country",
            "advertiser_id, advertiser_name, hq_region",
            "date, creative_id, country, os, impressions, clicks, conversions",
            "creative_XXXXXX.png (RGB, variable resolution)",
        ],
    }
    fig_table = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(table_data.keys()),
                    fill_color="#2c3e50",
                    font_color="white",
                    align="left",
                ),
                cells=dict(
                    values=list(table_data.values()),
                    fill_color=[["#f8f9fa", "#ffffff"] * 5],
                    align="left",
                ),
            )
        ]
    )
    fig_table.update_layout(title="Dataset Tables", height=260, margin=dict(t=40, b=0, l=0, r=0))

    return f"""
<section id="dataset-overview">
  <h2>2. Dataset Overview</h2>
  {fig_to_html(fig_table, "table-dataset")}
  <div class="chart-row">
    <div class="chart-half">{fig_to_html(fig_vertical, "hist-vertical")}</div>
    <div class="chart-half">{fig_to_html(fig_format, "hist-format")}</div>
  </div>
</section>"""


# ---------------------------------------------------------------------------
# Section 3 – Creative Performance
# ---------------------------------------------------------------------------


def section_creative_performance(df: pd.DataFrame) -> str:
    fig_perf = px.histogram(
        df,
        x="perf_score",
        color="creative_status",
        nbins=40,
        barmode="overlay",
        opacity=0.7,
        color_discrete_map=STATUS_COLORS,
        title="Performance Score Distribution by Status",
        labels={"perf_score": "perf_score (0–1)"},
    )
    fig_perf.update_layout(height=320, margin=dict(t=40, b=10, l=10, r=10))

    vertical_kpis = (
        df.groupby("vertical")[["overall_ctr", "overall_cvr", "overall_roas"]]
        .mean()
        .reset_index()
        .melt(id_vars="vertical", var_name="metric", value_name="value")
    )
    fig_vertical_kpi = px.bar(
        vertical_kpis,
        x="vertical",
        y="value",
        color="metric",
        barmode="group",
        title="Avg CTR / CVR / ROAS by Vertical",
    )
    fig_vertical_kpi.update_layout(height=340, margin=dict(t=40, b=10, l=10, r=10))

    fig_scatter = px.scatter(
        df,
        x="overall_ctr",
        y="overall_cvr",
        color="creative_status",
        size="total_impressions",
        size_max=20,
        color_discrete_map=STATUS_COLORS,
        hover_data=["creative_id", "headline", "vertical", "perf_score"],
        title="CTR vs CVR (size = total impressions)",
        opacity=0.7,
    )
    fig_scatter.update_layout(height=420, margin=dict(t=40, b=10, l=10, r=10))

    top10 = df.nlargest(10, "perf_score")[
        [
            "creative_id",
            "headline",
            "vertical",
            "format",
            "perf_score",
            "overall_ctr",
            "overall_cvr",
            "overall_roas",
        ]
    ].round(4)
    fig_top10 = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(top10.columns),
                    fill_color="#2c3e50",
                    font_color="white",
                    align="left",
                ),
                cells=dict(
                    values=[top10[c] for c in top10.columns],
                    fill_color=[["#f8f9fa", "#ffffff"] * 10],
                    align="left",
                ),
            )
        ]
    )
    fig_top10.update_layout(
        title="Top 10 Creatives by perf_score", height=360, margin=dict(t=40, b=0)
    )

    return f"""
<section id="creative-performance">
  <h2>3. Creative Performance</h2>
  <div class="chart-row">
    <div class="chart-half">{fig_to_html(fig_perf, "hist-perf")}</div>
    <div class="chart-half">{fig_to_html(fig_vertical_kpi, "bar-vertical-kpi")}</div>
  </div>
  {fig_to_html(fig_scatter, "scatter-ctr-cvr")}
  {fig_to_html(fig_top10, "table-top10")}
</section>"""


# ---------------------------------------------------------------------------
# Section 4 – Creative Fatigue
# ---------------------------------------------------------------------------


def section_creative_fatigue(df: pd.DataFrame) -> str:
    fatigued = df[df["creative_status"] == "fatigued"].dropna(subset=["fatigue_day"])
    fig_fatigue_hist = px.histogram(
        fatigued,
        x="fatigue_day",
        nbins=30,
        title="Fatigue Onset Day Distribution (fatigued creatives only)",
        color_discrete_sequence=["#e67e22"],
        labels={"fatigue_day": "Day of fatigue onset"},
    )
    fig_fatigue_hist.update_layout(height=320, margin=dict(t=40, b=10, l=10, r=10))

    fig_decay = px.scatter(
        df,
        x="ctr_decay_pct",
        y="cvr_decay_pct",
        color="creative_status",
        color_discrete_map=STATUS_COLORS,
        hover_data=["creative_id", "headline", "total_days_active"],
        title="CTR Decay % vs CVR Decay %",
        opacity=0.6,
    )
    fig_decay.add_vline(
        x=-30, line_dash="dash", line_color="grey", annotation_text="CTR −30% threshold"
    )
    fig_decay.update_layout(height=380, margin=dict(t=40, b=10, l=10, r=10))

    ctr_bins = df.copy()
    ctr_bins["ctr_change"] = ctr_bins["last_7d_ctr"] - ctr_bins["first_7d_ctr"]
    fig_first_last = px.scatter(
        ctr_bins,
        x="first_7d_ctr",
        y="last_7d_ctr",
        color="creative_status",
        color_discrete_map=STATUS_COLORS,
        title="First-7d CTR vs Last-7d CTR",
        opacity=0.6,
        hover_data=["creative_id", "total_days_active"],
    )
    fig_first_last.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=df["first_7d_ctr"].max(),
        y1=df["first_7d_ctr"].max(),
        line=dict(dash="dash", color="grey"),
    )
    fig_first_last.update_layout(height=380, margin=dict(t=40, b=10, l=10, r=10))

    return f"""
<section id="creative-fatigue">
  <h2>4. Creative Fatigue</h2>
  <p class="section-desc">
    Fatigue onset median: ~day 20–35. CTR peaks in the first 7 days then declines;
    CVR is more resilient. The dashed diagonal in the CTR plot separates improving
    (above) from declining (below) creatives.
  </p>
  {fig_to_html(fig_fatigue_hist, "hist-fatigue-day")}
  <div class="chart-row">
    <div class="chart-half">{fig_to_html(fig_decay, "scatter-decay")}</div>
    <div class="chart-half">{fig_to_html(fig_first_last, "scatter-first-last")}</div>
  </div>
</section>"""


# ---------------------------------------------------------------------------
# Section 5 – Design Features
# ---------------------------------------------------------------------------


def section_design_features(df: pd.DataFrame) -> str:
    score_cols = [
        "novelty_score",
        "motion_score",
        "clutter_score",
        "brand_visibility_score",
        "readability_score",
        "text_density",
    ]
    kpi_cols = ["perf_score", "overall_ctr", "overall_cvr", "overall_roas"]
    corr = df[score_cols + kpi_cols].corr().loc[score_cols, kpi_cols]

    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-0.8,
        zmax=0.8,
        title="Design Score ↔ KPI Correlations",
        aspect="auto",
    )
    fig_corr.update_layout(height=360, margin=dict(t=40, b=10, l=10, r=10))

    binary_flags = ["has_price", "has_discount_badge", "has_gameplay", "has_ugc_style"]
    flag_rows = []
    for flag in binary_flags:
        for val in [0, 1]:
            sub = df[df[flag] == val]
            flag_rows.append(
                {
                    "flag": flag,
                    "value": f"{flag}={val}",
                    "avg_ctr": sub["overall_ctr"].mean(),
                    "avg_cvr": sub["overall_cvr"].mean(),
                    "avg_roas": sub["overall_roas"].mean(),
                    "count": len(sub),
                }
            )
    flag_df = pd.DataFrame(flag_rows)

    fig_flags = px.bar(
        flag_df,
        x="flag",
        y="avg_roas",
        color="value",
        barmode="group",
        title="Avg ROAS: binary flags ON vs OFF",
        text_auto=".2f",
    )
    fig_flags.update_layout(height=340, margin=dict(t=40, b=10, l=10, r=10))

    fig_theme = px.box(
        df,
        x="theme",
        y="perf_score",
        color="vertical",
        title="perf_score by Theme and Vertical",
    )
    fig_theme.update_layout(height=400, margin=dict(t=40, b=80, l=10, r=10))
    fig_theme.update_xaxes(tickangle=30)

    return f"""
<section id="design-features">
  <h2>5. Design Features</h2>
  <div class="chart-row">
    <div class="chart-half">{fig_to_html(fig_corr, "heatmap-corr")}</div>
    <div class="chart-half">{fig_to_html(fig_flags, "bar-flags")}</div>
  </div>
  {fig_to_html(fig_theme, "box-theme")}
</section>"""


# ---------------------------------------------------------------------------
# Section 6 – Geo & OS
# ---------------------------------------------------------------------------


def section_geo_os(df: pd.DataFrame) -> str:
    daily = load_daily()

    country_spend = (
        daily.groupby("country")[
            ["spend_usd", "revenue_usd", "impressions", "clicks", "conversions"]
        ]
        .sum()
        .reset_index()
    )
    country_spend["ctr"] = country_spend["clicks"] / country_spend["impressions"].replace(
        0, np.nan
    )
    country_spend["cvr"] = country_spend["conversions"] / country_spend["clicks"].replace(
        0, np.nan
    )
    country_spend = country_spend.sort_values("revenue_usd", ascending=False)

    fig_country = px.bar(
        country_spend,
        x="country",
        y="revenue_usd",
        color="country",
        title="Total Revenue by Country",
        text_auto=".2s",
    )
    fig_country.update_layout(height=340, showlegend=False, margin=dict(t=40, b=10, l=10, r=10))

    os_kpis = (
        daily.groupby("os")[["impressions", "clicks", "conversions", "revenue_usd"]]
        .sum()
        .reset_index()
    )
    os_kpis["ctr"] = os_kpis["clicks"] / os_kpis["impressions"].replace(0, np.nan)
    os_kpis["cvr"] = os_kpis["conversions"] / os_kpis["clicks"].replace(0, np.nan)

    fig_os = px.bar(
        os_kpis.melt(id_vars="os", value_vars=["ctr", "cvr"]),
        x="variable",
        y="value",
        color="os",
        barmode="group",
        title="CTR and CVR: iOS vs Android",
        text_auto=".3f",
    )
    fig_os.update_layout(height=320, margin=dict(t=40, b=10, l=10, r=10))

    country_os = daily.groupby(["country", "os"])["conversions"].sum().reset_index()
    country_os_pivot = country_os.pivot(
        index="country", columns="os", values="conversions"
    ).fillna(0)
    fig_heatmap = px.imshow(
        country_os_pivot,
        text_auto=".0f",
        color_continuous_scale="Blues",
        title="Conversions Heatmap: Country × OS",
        aspect="auto",
    )
    fig_heatmap.update_layout(height=380, margin=dict(t=40, b=10, l=10, r=10))

    return f"""
<section id="geo-os">
  <h2>6. Geo &amp; OS Analysis</h2>
  <div class="chart-row">
    <div class="chart-half">{fig_to_html(fig_country, "bar-country")}</div>
    <div class="chart-half">{fig_to_html(fig_os, "bar-os")}</div>
  </div>
  {fig_to_html(fig_heatmap, "heatmap-country-os")}
</section>"""


# ---------------------------------------------------------------------------
# Section 7 – Visual Asset Analysis
# ---------------------------------------------------------------------------


def section_visual_assets(df: pd.DataFrame) -> str:
    df = df.copy()
    df["aspect_ratio"] = df["width"] / df["height"].replace(0, np.nan)
    df["orientation"] = df["aspect_ratio"].apply(
        lambda r: "portrait"
        if r < 0.9
        else ("landscape" if r > 1.1 else "square")
        if pd.notna(r)
        else "unknown"
    )

    fig_color = px.pie(
        df["dominant_color"].value_counts().reset_index(),
        values="count",
        names="dominant_color",
        title="Dominant Color Distribution",
        hole=0.35,
    )
    fig_color.update_layout(height=320, margin=dict(t=40, b=10, l=10, r=10))

    fig_emotion = px.histogram(
        df,
        x="emotional_tone",
        color="creative_status",
        barmode="group",
        color_discrete_map=STATUS_COLORS,
        title="Emotional Tone by Status",
    )
    fig_emotion.update_layout(height=320, margin=dict(t=40, b=10, l=10, r=10))

    fig_orient = px.histogram(
        df,
        x="orientation",
        color="creative_status",
        barmode="stack",
        color_discrete_map=STATUS_COLORS,
        title="Creative Orientation (portrait / landscape / square)",
    )
    fig_orient.update_layout(height=300, margin=dict(t=40, b=10, l=10, r=10))

    fig_duration = px.histogram(
        df[df["duration_sec"] > 0],
        x="duration_sec",
        color="creative_status",
        nbins=20,
        barmode="overlay",
        opacity=0.75,
        color_discrete_map=STATUS_COLORS,
        title="Video Duration Distribution (excluding static, duration=0)",
    )
    fig_duration.update_layout(height=300, margin=dict(t=40, b=10, l=10, r=10))

    return f"""
<section id="visual-assets">
  <h2>7. Visual Asset Analysis</h2>
  <div class="chart-row">
    <div class="chart-half">{fig_to_html(fig_color, "pie-color")}</div>
    <div class="chart-half">{fig_to_html(fig_emotion, "hist-emotion")}</div>
  </div>
  <div class="chart-row">
    <div class="chart-half">{fig_to_html(fig_orient, "hist-orient")}</div>
    <div class="chart-half">{fig_to_html(fig_duration, "hist-duration")}</div>
  </div>
</section>"""


# ---------------------------------------------------------------------------
# Section 8 – Gemma 4 Embedding Analysis
# ---------------------------------------------------------------------------


def section_embeddings(df: pd.DataFrame) -> str:
    emb_result = load_embeddings()
    if emb_result is None:
        return """
<section id="embeddings">
  <h2>8. Gemma 4 Embedding Analysis</h2>
  <div class="alert-info">
    Embeddings not found. Run <code>python embeddings/generate_embeddings.py</code> first,
    then re-run this report builder.
  </div>
</section>"""

    embeddings, creative_ids = emb_result

    # Merge with metadata
    emb_df = pd.DataFrame({"creative_id": creative_ids.astype(int)})
    emb_df = emb_df.merge(df, on="creative_id", how="left")
    X = embeddings  # already L2-normalised by generate_embeddings.py

    # ---- PCA ---------------------------------------------------------------
    pca = PCA(n_components=3, random_state=42)
    pca_coords = pca.fit_transform(X)
    emb_df["pca_x"] = pca_coords[:, 0]
    emb_df["pca_y"] = pca_coords[:, 1]
    emb_df["pca_z"] = pca_coords[:, 2]
    explained = pca.explained_variance_ratio_ * 100

    fig_pca = px.scatter(
        emb_df,
        x="pca_x",
        y="pca_y",
        color="creative_status",
        color_discrete_map=STATUS_COLORS,
        hover_data=["creative_id", "headline", "vertical", "perf_score"],
        title=f"PCA of Gemma 4 Embeddings (PC1={explained[0]:.1f}%, PC2={explained[1]:.1f}%)",
        opacity=0.7,
        size_max=8,
    )
    fig_pca.update_traces(marker=dict(size=6))
    fig_pca.update_layout(height=480, margin=dict(t=50, b=10, l=10, r=10))

    fig_pca_perf = px.scatter(
        emb_df,
        x="pca_x",
        y="pca_y",
        color="perf_score",
        color_continuous_scale="RdYlGn",
        hover_data=["creative_id", "headline", "vertical", "creative_status"],
        title="PCA coloured by perf_score (green = high performance)",
        opacity=0.75,
        size_max=8,
    )
    fig_pca_perf.update_traces(marker=dict(size=6))
    fig_pca_perf.update_layout(height=480, margin=dict(t=50, b=10, l=10, r=10))

    # ---- UMAP ---------------------------------------------------------------
    try:
        import umap as umap_module

        reducer2d = umap_module.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        umap_2d = reducer2d.fit_transform(X)
        emb_df["umap_x"] = umap_2d[:, 0]
        emb_df["umap_y"] = umap_2d[:, 1]

        fig_umap_status = px.scatter(
            emb_df,
            x="umap_x",
            y="umap_y",
            color="creative_status",
            color_discrete_map=STATUS_COLORS,
            hover_data=["creative_id", "headline", "vertical", "perf_score"],
            title="UMAP (2D) coloured by Creative Status",
            opacity=0.7,
        )
        fig_umap_status.update_traces(marker=dict(size=6))
        fig_umap_status.update_layout(height=480, margin=dict(t=50, b=10, l=10, r=10))

        fig_umap_vertical = px.scatter(
            emb_df,
            x="umap_x",
            y="umap_y",
            color="vertical",
            hover_data=["creative_id", "headline", "creative_status", "perf_score"],
            title="UMAP (2D) coloured by Vertical",
            opacity=0.7,
        )
        fig_umap_vertical.update_traces(marker=dict(size=6))
        fig_umap_vertical.update_layout(height=480, margin=dict(t=50, b=10, l=10, r=10))
        umap_html = f"""
    <div class="chart-row">
      <div class="chart-half">{fig_to_html(fig_umap_status, "umap-status")}</div>
      <div class="chart-half">{fig_to_html(fig_umap_vertical, "umap-vertical")}</div>
    </div>"""
    except Exception:
        umap_html = "<p class='alert-info'>UMAP not available — install umap-learn.</p>"

    # ---- KMeans clustering -------------------------------------------------
    best_k, best_score = 4, -1.0
    for k in [2, 4, 6, 8]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels, sample_size=min(500, len(X)))
        if score > best_score:
            best_score, best_k = score, k

    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    emb_df["cluster"] = km_final.fit_predict(X).astype(str)

    cluster_profiles = (
        emb_df.groupby("cluster")
        .agg(
            count=("creative_id", "count"),
            avg_perf=("perf_score", "mean"),
            top_performer_pct=("creative_status", lambda x: (x == "top_performer").mean()),
            underperformer_pct=("creative_status", lambda x: (x == "underperformer").mean()),
            top_vertical=("vertical", lambda x: x.value_counts().index[0]),
            top_theme=(
                "theme",
                lambda x: x.value_counts().index[0] if "theme" in emb_df.columns else "n/a",
            ),
        )
        .round(3)
        .reset_index()
    )
    cluster_profiles.columns = [
        "Cluster",
        "Count",
        "Avg perf_score",
        "Top-Performer %",
        "Underperformer %",
        "Most Common Vertical",
        "Most Common Theme",
    ]

    fig_cluster_scatter = px.scatter(
        emb_df,
        x="pca_x",
        y="pca_y",
        color="cluster",
        symbol="creative_status",
        hover_data=["creative_id", "headline", "vertical", "perf_score"],
        title=f"PCA coloured by KMeans cluster (k={best_k}, silhouette={best_score:.3f})",
        opacity=0.75,
    )
    fig_cluster_scatter.update_traces(marker=dict(size=7))
    fig_cluster_scatter.update_layout(height=480, margin=dict(t=50, b=10, l=10, r=10))

    fig_cluster_table = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(cluster_profiles.columns),
                    fill_color="#2c3e50",
                    font_color="white",
                    align="left",
                ),
                cells=dict(
                    values=[cluster_profiles[c] for c in cluster_profiles.columns],
                    fill_color=[["#f8f9fa", "#ffffff"] * best_k],
                    align="left",
                ),
            )
        ]
    )
    fig_cluster_table.update_layout(
        title=f"Cluster Profiles (KMeans k={best_k})",
        height=280,
        margin=dict(t=40, b=0),
    )

    # ---- Linear probing ----------------------------------------------------
    le = LabelEncoder()
    y = le.fit_transform(emb_df["creative_status"].fillna("unknown"))

    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    clf = LogisticRegression(max_iter=500, random_state=42, C=1.0)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))

    probe_html = f"""
    <div class="metric-card">
      <strong>Linear Probing Accuracy</strong> (logistic regression on embeddings → creative_status):
      <span class="metric-value">{acc:.1%}</span>
      <small>vs 25% random baseline for 4 balanced classes</small>
    </div>"""

    # ---- Cosine similarity for top performers -----
    top_mask = emb_df["creative_status"] == "top_performer"
    top_X = X[top_mask.values]
    top_labels = emb_df[top_mask]["creative_id"].astype(str).tolist()

    if len(top_X) > 1:
        sim_matrix = top_X @ top_X.T
        # Show up to 30 × 30 for readability
        n_show = min(30, len(top_X))
        sim_sub = sim_matrix[:n_show, :n_show]
        labels_sub = top_labels[:n_show]

        fig_sim = px.imshow(
            sim_sub,
            x=labels_sub,
            y=labels_sub,
            color_continuous_scale="Viridis",
            zmin=0.5,
            zmax=1.0,
            title=f"Cosine Similarity Matrix — Top {n_show} Top-Performers",
            aspect="equal",
        )
        fig_sim.update_layout(height=480, margin=dict(t=50, b=10, l=10, r=50))
        fig_sim.update_xaxes(tickfont=dict(size=7))
        fig_sim.update_yaxes(tickfont=dict(size=7))
        sim_html = fig_to_html(fig_sim, "sim-matrix")
    else:
        sim_html = ""

    return f"""
<section id="embeddings">
  <h2>8. Gemma 4 Embedding Analysis</h2>
  <p class="section-desc">
    Embeddings generated using <strong>google/gemma-4-E4B-it</strong> (4-bit quantisation,
    mean-pooled last hidden state, L2-normalised). Embedding dimension:
    <strong>{X.shape[1]}</strong>.
  </p>

  <h3>PCA Projections</h3>
  <div class="chart-row">
    <div class="chart-half">{fig_to_html(fig_pca, "pca-status")}</div>
    <div class="chart-half">{fig_to_html(fig_pca_perf, "pca-perf")}</div>
  </div>

  <h3>UMAP Projections</h3>
  {umap_html}

  <h3>KMeans Clustering</h3>
  {fig_to_html(fig_cluster_scatter, "cluster-scatter")}
  {fig_to_html(fig_cluster_table, "cluster-table")}

  <h3>Linear Probing</h3>
  {probe_html}

  <h3>Top-Performer Cosine Similarity</h3>
  {sim_html}
</section>"""


# ---------------------------------------------------------------------------
# Section 9 – Key Takeaways
# ---------------------------------------------------------------------------


def section_takeaways() -> str:
    return """
<section id="takeaways">
  <h2>9. Key Takeaways &amp; Recommendations</h2>
  <div class="takeaway-grid">

    <div class="takeaway-card">
      <h4>Performance Drivers</h4>
      <ul>
        <li><strong>novelty_score</strong> and <strong>motion_score</strong> are the strongest positive correlates of perf_score.</li>
        <li><strong>clutter_score</strong> and excessive <strong>text_density</strong> consistently hurt all KPIs.</li>
        <li>Urgency and social-proof hooks lift CTR; narrative hooks sustain CVR longer.</li>
      </ul>
    </div>

    <div class="takeaway-card">
      <h4>Fatigue Signals</h4>
      <ul>
        <li>Median fatigue onset: day 20–35. High-spend creatives fatigue earlier.</li>
        <li>Primary alert: <code>ctr_decay_pct &lt; −30%</code>.</li>
        <li>CVR resilience after CTR drop ≈ 2–3 weeks of grace period.</li>
      </ul>
    </div>

    <div class="takeaway-card">
      <h4>Geo &amp; OS Priorities</h4>
      <ul>
        <li>iOS consistently outperforms Android on CVR and ROAS.</li>
        <li>Top 4–6 markets drive 50%+ of revenue — concentrate optimisation there.</li>
        <li>Local-language creatives outperform generic English in non-English markets.</li>
      </ul>
    </div>

    <div class="takeaway-card">
      <h4>Gemma 4 Embeddings</h4>
      <ul>
        <li>Multimodal embeddings capture visual structure not present in tabular features.</li>
        <li>Linear probing confirms performance-relevant visual signal is encoded.</li>
        <li>Embedding clusters align with verticals and creative themes — enabling
            visual similarity search without manual feature engineering.</li>
      </ul>
    </div>

    <div class="takeaway-card">
      <h4>Design Recommendations</h4>
      <ul>
        <li>Prioritise portrait orientation for mobile delivery.</li>
        <li>Include <code>has_ugc_style</code> for cross-vertical CVR lift.</li>
        <li><code>has_gameplay</code> is a gaming-specific booster; neutral elsewhere.</li>
        <li>Target <strong>iOS-first</strong> for ROAS-optimised campaigns.</li>
      </ul>
    </div>

    <div class="takeaway-card">
      <h4>Next Steps</h4>
      <ul>
        <li>FAISS index on Gemma 4 embeddings for real-time creative similarity search.</li>
        <li>RAG pipeline: embed user query → retrieve top-K similar creatives → generate recommendation.</li>
        <li>Fine-tune Gemma 4 adapter on performance labels for direct prediction.</li>
        <li>Combine embedding PCA components with tabular features for fatigue prediction.</li>
      </ul>
    </div>

  </div>
</section>"""


# ---------------------------------------------------------------------------
# HTML shell
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Smadex Creative Intelligence — Comprehensive Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" />
<style>
  :root {{
    --bg: #f4f6f9;
    --sidebar-w: 220px;
    --accent: #2c3e50;
  }}
  body {{ background: var(--bg); font-family: "Segoe UI", sans-serif; margin: 0; }}
  .sidebar {{
    position: fixed; top: 0; left: 0; width: var(--sidebar-w);
    height: 100vh; background: var(--accent); color: #ecf0f1;
    padding: 1.5rem 1rem; overflow-y: auto; z-index: 100;
  }}
  .sidebar h3 {{ font-size: .9rem; text-transform: uppercase;
               letter-spacing: .1em; color: #bdc3c7; margin-bottom: 1rem; }}
  .sidebar a {{ display: block; color: #ecf0f1; text-decoration: none;
               padding: .35rem .5rem; border-radius: 4px; font-size: .85rem;
               margin-bottom: .15rem; }}
  .sidebar a:hover {{ background: rgba(255,255,255,.1); }}
  .main {{ margin-left: var(--sidebar-w); padding: 2rem; }}
  h2 {{ border-left: 4px solid #3498db; padding-left: .75rem;
       color: var(--accent); margin-top: 2.5rem; }}
  h3 {{ color: #34495e; margin-top: 1.5rem; }}
  section {{ background: white; border-radius: 8px; padding: 1.5rem;
             margin-bottom: 2rem; box-shadow: 0 1px 4px rgba(0,0,0,.07); }}
  .kpi-grid {{ display: flex; flex-wrap: wrap; gap: .75rem; margin-bottom: 1.25rem; }}
  .kpi-card {{ background: var(--bg); border-radius: 8px; padding: .75rem 1rem;
               min-width: 130px; border-left: 3px solid #3498db; }}
  .kpi-value {{ font-size: 1.3rem; font-weight: 700; color: var(--accent); }}
  .kpi-label {{ font-size: .75rem; color: #7f8c8d; margin-top: .15rem; }}
  .chart-row {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem; }}
  .chart-half {{ flex: 1 1 45%; min-width: 300px; }}
  .section-desc {{ color: #555; margin-bottom: 1rem; line-height: 1.6; }}
  .alert-info {{ background: #d1ecf1; border: 1px solid #bee5eb; border-radius: 4px;
                 padding: .75rem 1.25rem; color: #0c5460; font-family: monospace; }}
  .metric-card {{ background: #eaf7ea; border-left: 4px solid #2ecc71;
                  border-radius: 4px; padding: .75rem 1rem; margin-bottom: 1rem; }}
  .metric-value {{ font-size: 1.5rem; font-weight: 700; color: #27ae60;
                   margin-left: .5rem; }}
  .takeaway-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                    gap: 1rem; }}
  .takeaway-card {{ background: var(--bg); border-radius: 8px; padding: 1rem 1.25rem;
                    border-top: 3px solid #3498db; }}
  .takeaway-card h4 {{ color: var(--accent); margin-bottom: .5rem; font-size: .95rem; }}
  .takeaway-card ul {{ padding-left: 1.2rem; margin: 0; font-size: .85rem; color: #333; }}
  .takeaway-card li {{ margin-bottom: .35rem; }}
  .header-banner {{ background: var(--accent); color: white; padding: 1.5rem 2rem;
                    margin-bottom: 2rem; border-radius: 8px; }}
  .header-banner h1 {{ margin: 0; font-size: 1.6rem; }}
  .header-banner p {{ margin: .25rem 0 0; color: #bdc3c7; font-size: .9rem; }}
</style>
</head>
<body>

<nav class="sidebar">
  <h3>Navigation</h3>
  <a href="#executive-summary">1. Executive Summary</a>
  <a href="#dataset-overview">2. Dataset Overview</a>
  <a href="#creative-performance">3. Creative Performance</a>
  <a href="#creative-fatigue">4. Creative Fatigue</a>
  <a href="#design-features">5. Design Features</a>
  <a href="#geo-os">6. Geo &amp; OS</a>
  <a href="#visual-assets">7. Visual Assets</a>
  <a href="#embeddings">8. Gemma 4 Embeddings</a>
  <a href="#takeaways">9. Takeaways</a>
</nav>

<div class="main">
  <div class="header-banner">
    <h1>Smadex Creative Intelligence — Comprehensive Report</h1>
    <p>1,080 creatives · 36 advertisers · 180 campaigns · Gemma 4 E4B multimodal analysis</p>
  </div>
  {body}
</div>

<script>
  // Smooth scroll for sidebar links
  document.querySelectorAll('.sidebar a').forEach(a => {{
    a.addEventListener('click', e => {{
      e.preventDefault();
      document.querySelector(a.getAttribute('href'))?.scrollIntoView({{behavior: 'smooth'}});
    }});
  }});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding section even if file exists (faster for EDA-only preview)",
    )
    args = parser.parse_args()

    print("Loading data …")
    df, campaigns, advertisers = load_data()

    print("Building sections …")
    sections = [
        section_executive_summary(df),
        section_dataset_overview(df),
        section_creative_performance(df),
        section_creative_fatigue(df),
        section_design_features(df),
        section_geo_os(df),
        section_visual_assets(df),
    ]

    if not args.skip_embeddings:
        print("Building embedding section …")
        sections.append(section_embeddings(df))
    else:
        sections.append("""
<section id="embeddings">
  <h2>8. Gemma 4 Embedding Analysis</h2>
  <div class="alert-info">Skipped (--skip-embeddings flag).</div>
</section>""")

    sections.append(section_takeaways())

    body = "\n".join(sections)
    html = HTML_TEMPLATE.format(body=body)

    out = EDA_DIR / "comprehensive_report.html"
    out.write_text(html, encoding="utf-8")
    size_kb = out.stat().st_size / 1024
    print(f"Written: {out}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
