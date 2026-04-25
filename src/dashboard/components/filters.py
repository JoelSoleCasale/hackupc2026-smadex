"""Sidebar filter widgets for the Smadex Creative Intelligence dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from loguru import logger

# ---------------------------------------------------------------------------
# Region → ISO-2 country code mappings
# ---------------------------------------------------------------------------
REGION_COUNTRIES: dict[str, list[str]] = {
    "Europe": [
        "DE",
        "FR",
        "ES",
        "IT",
        "NL",
        "PL",
        "SE",
        "NO",
        "DK",
        "FI",
        "BE",
        "AT",
        "CH",
        "PT",
        "IE",
        "GB",
    ],
    "Americas": ["US", "CA", "MX", "BR", "AR", "CO", "CL", "PE"],
    "APAC": [
        "JP",
        "KR",
        "AU",
        "NZ",
        "SG",
        "IN",
        "TH",
        "ID",
        "PH",
        "VN",
        "MY",
        "HK",
        "TW",
    ],
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_filters(summary_df: pd.DataFrame, daily_df: pd.DataFrame) -> dict:
    """Render all sidebar filter widgets and return a filter state dict.

    Parameters
    ----------
    summary_df:
        DataFrame from ``src.data.loader.load_creative_summary()``.  Expected
        columns: advertiser_name, campaign_id, creative_id, format, vertical,
        perf_score, overall_ctr, overall_cvr, overall_roas.
    daily_df:
        DataFrame from `load_daily_stats()` to get valid countries.


    Returns
    -------
    dict with keys: advertiser, metric, campaigns, creatives, countries, os, formats.
        - countries: Empty list means all countries (no filter applied).
    """
    logger.debug("Rendering sidebar filters")

    # ------------------------------------------------------------------
    # 1. Advertiser selector
    # ------------------------------------------------------------------
    advertiser_options = sorted(summary_df["advertiser_name"].unique())
    advertiser = st.sidebar.selectbox("Advertiser", advertiser_options)

    # ------------------------------------------------------------------
    # 2. Compare metric
    # ------------------------------------------------------------------
    metric = st.sidebar.selectbox(
        "Compare metric",
        ["perf_score", "ctr", "cvr", "roas"],
        index=0,
    )

    # ------------------------------------------------------------------
    # 3. Campaigns & Creatives (File Tree)
    # ------------------------------------------------------------------
    adv_df = summary_df[summary_df["advertiser_name"] == advertiser]

    st.sidebar.markdown("**Campaigns & Creatives**")

    selected_campaigns = []
    selected_creatives = []

    # Initialize all checkboxes to True in session state if not set
    # We want default to be all selected
    for camp_id in sorted(adv_df["campaign_id"].unique()):
        camp_creatives = sorted(adv_df[adv_df["campaign_id"] == camp_id]["creative_id"].unique())
        creative_keys = [f"chk_{advertiser}_{camp_id}_{creat_id}" for creat_id in camp_creatives]

        for st_key in creative_keys:
            if st_key not in st.session_state:
                st.session_state[st_key] = True

        selected_in_campaign = sum(
            1 for st_key in creative_keys if st.session_state.get(st_key, False)
        )
        campaign_label = f"📁 {camp_id} ({selected_in_campaign}/{len(camp_creatives)})"

        with st.sidebar.expander(campaign_label, expanded=False):
            col_select, col_deselect = st.columns(2)
            with col_select:
                if st.button(
                    "Select all",
                    key=f"btn_select_all_{advertiser}_{camp_id}",
                    use_container_width=True,
                ):
                    for st_key in creative_keys:
                        st.session_state[st_key] = True
                    st.rerun()
            with col_deselect:
                if st.button(
                    "Deselect all",
                    key=f"btn_deselect_all_{advertiser}_{camp_id}",
                    use_container_width=True,
                ):
                    for st_key in creative_keys:
                        st.session_state[st_key] = False
                    st.rerun()

            for creat_id in camp_creatives:
                st_key = f"chk_{advertiser}_{camp_id}_{creat_id}"
                checked = st.checkbox(f"📄 {creat_id}", value=st.session_state[st_key], key=st_key)
                if checked:
                    selected_creatives.append(creat_id)

        if any(st.session_state.get(st_key, False) for st_key in creative_keys):
            selected_campaigns.append(camp_id)

    campaigns = selected_campaigns
    creatives = selected_creatives

    # ------------------------------------------------------------------
    # 4. Country selector (File Tree)
    # ------------------------------------------------------------------
    st.sidebar.markdown("**Countries**")

    # Get all unique valid countries
    valid_countries = sorted(daily_df["country"].dropna().unique())
    selected_countries = []

    all_regions = ["Europe", "Americas", "APAC"]

    for r_label in all_regions:
        r_countries = sorted([c for c in REGION_COUNTRIES[r_label] if c in valid_countries])
        if not r_countries:
            continue

        region_keys = [f"chk_country_{c_code}" for c_code in r_countries]
        for st_key_c in region_keys:
            if st_key_c not in st.session_state:
                # Default to all selected; an empty final list still means no country filter.
                st.session_state[st_key_c] = True

        selected_in_region = sum(
            1 for st_key_c in region_keys if st.session_state.get(st_key_c, False)
        )
        region_label = f"🌍 {r_label} ({selected_in_region}/{len(r_countries)})"

        with st.sidebar.expander(region_label, expanded=False):
            col_select_region, col_deselect_region = st.columns(2)
            with col_select_region:
                if st.button(
                    "Select all",
                    key=f"btn_select_all_region_{r_label}",
                    use_container_width=True,
                ):
                    for st_key_c in region_keys:
                        st.session_state[st_key_c] = True
                    st.rerun()
            with col_deselect_region:
                if st.button(
                    "Deselect all",
                    key=f"btn_deselect_all_region_{r_label}",
                    use_container_width=True,
                ):
                    for st_key_c in region_keys:
                        st.session_state[st_key_c] = False
                    st.rerun()

            for c_code in r_countries:
                st_key_c = f"chk_country_{c_code}"

                checked_c = st.checkbox(
                    f"📍 {c_code}", value=st.session_state[st_key_c], key=st_key_c
                )
                if checked_c:
                    selected_countries.append(c_code)

    countries = selected_countries

    # ------------------------------------------------------------------
    # 6. OS
    # ------------------------------------------------------------------
    os_choice = st.sidebar.radio("OS", ["All", "Android", "iOS"], horizontal=True)

    # ------------------------------------------------------------------
    # 7. Format
    # ------------------------------------------------------------------
    format_options = sorted(summary_df["format"].unique().tolist())
    formats = st.sidebar.multiselect(
        "Format",
        options=format_options,
        default=format_options,
    )

    filter_state = {
        "advertiser": advertiser,
        "metric": metric,
        "campaigns": campaigns,
        "creatives": creatives,
        "countries": countries,
        "os": os_choice,
        "formats": formats,
    }

    applied_state_key = "performance_explorer_applied_filters"
    if applied_state_key not in st.session_state:
        st.session_state[applied_state_key] = filter_state

    if st.sidebar.button("Apply Filters", use_container_width=True, type="primary"):
        st.session_state[applied_state_key] = filter_state
        st.rerun()

    applied_filters = st.session_state[applied_state_key]
    logger.debug("Applied filter state: {}", applied_filters)
    return applied_filters
