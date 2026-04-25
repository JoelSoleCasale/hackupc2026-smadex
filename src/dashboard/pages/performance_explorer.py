import streamlit as st
from loguru import logger

from src.dashboard.components.comparison_charts import (
    render_distribution_box,
    render_format_bar,
    render_geo_heatmap,
    render_peer_rank_bar,
)
from src.dashboard.components.creative_strip import render_creative_strip
from src.dashboard.components.filters import render_filters
from src.dashboard.components.kpi_cards import render_kpi_cards
from src.data.loader import apply_filters, load_creative_summary, load_daily_stats


def _map_ids(summary_df, daily_df):
    """Map UUIDs to clean IDs like Campaign 1, Creative 1.1"""
    summary_df["campaign_id"] = summary_df["campaign_id"].astype(str)
    summary_df["creative_id"] = summary_df["creative_id"].astype(str)
    daily_df["campaign_id"] = daily_df["campaign_id"].astype(str)
    daily_df["creative_id"] = daily_df["creative_id"].astype(str)

    # Group by advertiser to ensure consistency
    for adv in summary_df["advertiser_name"].unique():
        adv_mask = summary_df["advertiser_name"] == adv
        campaigns = sorted(summary_df.loc[adv_mask, "campaign_id"].unique())

        for i, camp_id in enumerate(campaigns, 1):
            camp_name = f"Campaign {i}"
            camp_mask = adv_mask & (summary_df["campaign_id"] == camp_id)
            summary_df.loc[camp_mask, "campaign_id"] = camp_name
            daily_df.loc[daily_df["campaign_id"] == camp_id, "campaign_id"] = camp_name

            creatives = sorted(summary_df.loc[camp_mask, "creative_id"].unique())
            for j, creat_id in enumerate(creatives, 1):
                creat_name = f"Creative {i}.{j}"
                summary_df.loc[summary_df["creative_id"] == creat_id, "creative_id"] = creat_name
                daily_df.loc[daily_df["creative_id"] == creat_id, "creative_id"] = creat_name


def render_performance_explorer() -> None:
    st.title("Creative Performance Explorer")
    st.caption("Explore your creatives vs. sector peers. Use the sidebar to filter.")

    summary_df = load_creative_summary()
    daily_df = load_daily_stats()

    # Map IDs before filtering
    _map_ids(summary_df, daily_df)

    filters = render_filters(summary_df, daily_df)

    your_summary = apply_filters(summary_df, filters)
    your_daily = apply_filters(daily_df, filters)

    advertiser_rows = summary_df[summary_df["advertiser_name"] == filters["advertiser"]]
    if advertiser_rows.empty:
        st.warning("No data found for the selected advertiser.")
        return
    vertical = advertiser_rows["vertical"].iloc[0]

    sector_summary = summary_df[summary_df["vertical"] == vertical]
    peers_summary = sector_summary[sector_summary["advertiser_name"] != filters["advertiser"]]

    peer_creative_ids = peers_summary["creative_id"].tolist()
    peers_daily = apply_filters(
        daily_df[daily_df["creative_id"].isin(peer_creative_ids)],
        {
            "countries": filters["countries"],
            "os": filters["os"],
            "formats": filters["formats"],
        },
    )

    logger.debug(
        "Advertiser: {} | vertical: {} | your creatives: {} | peer creatives: {}",
        filters["advertiser"],
        vertical,
        len(your_summary),
        len(peers_summary),
    )

    render_kpi_cards(your_summary, your_daily, filters, peers_summary)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            render_peer_rank_bar(
                your_summary,
                your_daily,
                filters,
                vertical,
                peers_summary,
                peers_daily,
            ),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            render_geo_heatmap(your_summary, your_daily, filters, peers_summary, peers_daily),
            use_container_width=True,
        )

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(
            render_distribution_box(your_summary, your_daily, filters, peers_summary, peers_daily),
            use_container_width=True,
        )
    with col4:
        st.plotly_chart(
            render_format_bar(your_summary, your_daily, filters, peers_summary, peers_daily),
            use_container_width=True,
        )

    st.divider()

    render_creative_strip(your_summary, filters)
