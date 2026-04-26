import streamlit as st
from loguru import logger

from src.dashboard.components.campaign_cards import render_campaign_cards
from src.dashboard.components.comparison_charts import render_geo_heatmap, render_peer_rank_bar
from src.dashboard.components.kpi_cards import render_kpi_cards
from src.dashboard.pages.ad_detail_view import render_ad_detail_view
from src.dashboard.pages.campaign_view import render_campaign_view
from src.data.loader import load_campaigns, load_creative_summary, load_daily_stats


def _map_ids(summary_df, daily_df) -> dict:
    """Map UUIDs to clean IDs like Campaign 1, Creative 1.1.

    Returns the campaign UUID→name mapping so it can be applied to other DataFrames.
    """
    summary_df["campaign_id"] = summary_df["campaign_id"].astype(str)
    summary_df["creative_id"] = summary_df["creative_id"].astype(str)
    daily_df["campaign_id"] = daily_df["campaign_id"].astype(str)
    daily_df["creative_id"] = daily_df["creative_id"].astype(str)

    campaign_id_map: dict[str, str] = {}

    for adv in summary_df["advertiser_name"].unique():
        adv_mask = summary_df["advertiser_name"] == adv
        campaigns = sorted(summary_df.loc[adv_mask, "campaign_id"].unique())

        for i, camp_id in enumerate(campaigns, 1):
            camp_name = f"Campaign {i}"
            campaign_id_map[camp_id] = camp_name
            camp_mask = adv_mask & (summary_df["campaign_id"] == camp_id)
            summary_df.loc[camp_mask, "campaign_id"] = camp_name
            daily_df.loc[daily_df["campaign_id"] == camp_id, "campaign_id"] = camp_name

            creatives = sorted(summary_df.loc[camp_mask, "creative_id"].unique())
            for j, creat_id in enumerate(creatives, 1):
                creat_name = f"Creative {i}.{j}"
                summary_df.loc[summary_df["creative_id"] == creat_id, "creative_id"] = creat_name
                daily_df.loc[daily_df["creative_id"] == creat_id, "creative_id"] = creat_name

    return campaign_id_map


def _init_session_state() -> None:
    defaults = {
        "current_view": "overview",
        "selected_campaign": None,
        "selected_creative": None,
        "advertiser_choice": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def render_performance_explorer() -> None:
    _init_session_state()

    summary_df = load_creative_summary()
    daily_df = load_daily_stats()
    campaigns_df = load_campaigns()

    campaign_id_map = _map_ids(summary_df, daily_df)

    # Apply campaign ID mapping to campaigns_df
    campaigns_df["campaign_id"] = (
        campaigns_df["campaign_id"].astype(str).map(lambda x: campaign_id_map.get(x, x))
    )

    # --- Top bar: advertiser selector + metric selector ---
    advertiser_options = sorted(summary_df["advertiser_name"].unique())

    # Seed advertiser_choice on first load
    if st.session_state.advertiser_choice not in advertiser_options:
        st.session_state.advertiser_choice = advertiser_options[0]

    top_left, top_right = st.columns([3, 1])
    with top_left:
        # Use index= (not key=) so navigation reruns don't reset the widget
        advertiser = st.selectbox(
            "Advertiser",
            advertiser_options,
            index=advertiser_options.index(st.session_state.advertiser_choice),
        )
    with top_right:
        metric = st.selectbox(
            "Metric",
            ["perf_score", "ctr", "cvr", "roas"],
            key="selected_metric",
        )

    # Reset drill-down only when the user actively changes the advertiser
    if advertiser != st.session_state.advertiser_choice:
        st.session_state.advertiser_choice = advertiser
        st.session_state.current_view = "overview"
        st.session_state.selected_campaign = None
        st.session_state.selected_creative = None
    else:
        st.session_state.advertiser_choice = advertiser

    # Derive peer data (reused by overview and campaign view)
    advertiser_rows = summary_df[summary_df["advertiser_name"] == advertiser]
    if advertiser_rows.empty:
        st.warning("No data found for the selected advertiser.")
        return

    vertical = advertiser_rows["vertical"].iloc[0]
    sector_summary = summary_df[summary_df["vertical"] == vertical]
    peers_summary = sector_summary[sector_summary["advertiser_name"] != advertiser]

    logger.debug(
        "view={} advertiser={} vertical={} peers={}",
        st.session_state.current_view,
        advertiser,
        vertical,
        len(peers_summary),
    )

    # --- Router ---
    view = st.session_state.current_view

    if view == "overview":
        _render_overview(
            summary_df, daily_df, campaigns_df, peers_summary, advertiser, vertical, metric
        )

    elif view == "campaign":
        campaign_id = st.session_state.selected_campaign
        if not campaign_id:
            st.session_state.current_view = "overview"
            st.rerun()
        render_campaign_view(
            summary_df, daily_df, campaigns_df, peers_summary, advertiser, campaign_id
        )

    elif view == "ad_detail":
        campaign_id = st.session_state.selected_campaign
        creative_id = st.session_state.selected_creative
        if not creative_id:
            st.session_state.current_view = "campaign"
            st.rerun()
        render_ad_detail_view(summary_df, advertiser, campaign_id, creative_id, campaigns_df)


def _render_overview(
    summary_df, daily_df, campaigns_df, peers_summary, advertiser: str, vertical: str, metric: str
) -> None:
    st.title("Creative Performance Explorer")

    your_summary = summary_df[summary_df["advertiser_name"] == advertiser]
    your_creative_ids = your_summary["creative_id"].tolist()
    your_daily = daily_df[daily_df["creative_id"].isin(your_creative_ids)]

    peer_creative_ids = peers_summary["creative_id"].tolist()
    peers_daily = daily_df[daily_df["creative_id"].isin(peer_creative_ids)]

    filters = {"metric": metric, "advertiser": advertiser}

    render_kpi_cards(your_summary, your_daily, filters, peers_summary)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            render_peer_rank_bar(
                your_summary, your_daily, filters, vertical, peers_summary, peers_daily
            ),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            render_geo_heatmap(your_summary, your_daily, filters, peers_summary, peers_daily),
            use_container_width=True,
        )

    st.divider()

    adv_campaigns = campaigns_df[campaigns_df["advertiser_name"] == advertiser]
    render_campaign_cards(your_summary, adv_campaigns)
