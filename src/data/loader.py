"""
Data loader for the Smadex Creative Intelligence dashboard.

All paths are resolved relative to this file's location so the module
works regardless of the working directory the app is launched from.
"""

from pathlib import Path

import pandas as pd
import streamlit as st
from loguru import logger

# Project root: src/data/loader.py → up 3 levels
_DATA_DIR = Path(__file__).parent.parent.parent / "data"


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------


@st.cache_data
def load_creative_summary() -> pd.DataFrame:
    """Load creative_summary.csv merged with advertisers.csv metadata.

    Returns a 1 080-row DataFrame that includes all columns from
    creative_summary plus any *new* columns from advertisers (e.g.
    advertiser_id, hq_region).  The vertical column that already exists
    in creative_summary is kept as-is; duplicates from the merge are
    dropped automatically.
    """
    summary_path = _DATA_DIR / "creative_summary.csv"
    advertisers_path = _DATA_DIR / "advertisers.csv"

    logger.info("Loading creative_summary from {}", summary_path)
    summary = pd.read_csv(summary_path)
    logger.info("creative_summary shape: {}", summary.shape)

    logger.info("Loading advertisers from {}", advertisers_path)
    try:
        advertisers = pd.read_csv(advertisers_path)
    except FileNotFoundError:
        logger.warning(
            "advertisers.csv not found at {}; returning summary without merge", advertisers_path
        )
        return summary

    # Drop columns from advertisers that already exist in summary to avoid
    # _x / _y suffixes.  Always keep the join key (advertiser_name).
    duplicate_cols = [
        c for c in advertisers.columns if c != "advertiser_name" and c in summary.columns
    ]
    if duplicate_cols:
        logger.debug(
            "Dropping {} columns from advertisers to avoid duplicates: {}",
            len(duplicate_cols),
            duplicate_cols,
        )
        advertisers = advertisers.drop(columns=duplicate_cols)

    merged = summary.merge(advertisers, on="advertiser_name", how="left")
    logger.info("Merged creative_summary shape: {}", merged.shape)

    if len(merged) < len(summary):
        logger.warning("Merge resulted in row loss: {} → {} rows", len(summary), len(merged))

    return merged


@st.cache_data
def load_daily_stats() -> pd.DataFrame:
    """Load creative_daily_country_os_stats.csv with derived KPI columns.

    Derived columns (safe-divided, no ZeroDivisionError):
      ctr  = clicks / impressions
      cvr  = conversions / clicks
      roas = revenue_usd / spend_usd

    Returns ~192 k-row DataFrame.
    """
    path = _DATA_DIR / "creative_daily_country_os_stats.csv"
    logger.info("Loading daily stats from {}", path)

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    logger.info("Daily stats shape after load: {}", df.shape)

    # Safe-divide helpers (result is NaN when denominator is 0)
    df["ctr"] = df["clicks"].div(df["impressions"].replace(0, float("nan")))
    df["cvr"] = df["conversions"].div(df["clicks"].replace(0, float("nan")))
    df["roas"] = df["revenue_usd"].div(df["spend_usd"].replace(0, float("nan")))

    logger.info("Daily stats shape with derived cols: {}", df.shape)
    return df


@st.cache_data
def load_correlations(method: str = "statistical", metric: str = "perf_score") -> pd.DataFrame:
    """Load a pre-computed correlation parquet for a given method/metric combo.

    Available methods: "statistical", "rf_signed"
    Available metrics: "perf_score", "overall_ctr", "overall_cvr", "overall_roas"
    """
    path = _DATA_DIR / "correlations" / f"correlations_{method}_{metric}.parquet"
    logger.info("Loading correlations from {}", path)
    df = pd.read_parquet(path)
    logger.info("Correlations shape: {}", df.shape)
    return df


@st.cache_data
def load_campaigns() -> pd.DataFrame:
    """Load campaigns.csv with campaign-level metadata.

    Returns a 180-row DataFrame with columns including:
    campaign_id, advertiser_id, advertiser_name, app_name, vertical,
    objective, primary_theme, target_age_segment, target_os, countries,
    start_date, end_date, daily_budget_usd, kpi_goal.
    """
    path = _DATA_DIR / "campaigns.csv"
    logger.info("Loading campaigns from {}", path)
    df = pd.read_csv(path)
    logger.info("Campaigns shape: {}", df.shape)
    return df


# ---------------------------------------------------------------------------
# Filter utility
# ---------------------------------------------------------------------------


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Return a filtered copy of *df* based on keys present in *filters*.

    Works for both creative_summary and daily_stats schemas — filters are
    applied only when the corresponding column exists in *df*.

    Supported filter keys
    ---------------------
    "advertiser"  -> advertiser_name (str, exact match)
    "campaigns"   -> campaign_id     (list[int], isin; skipped if empty)
    "creatives"   -> creative_id     (list[int], isin; skipped if empty)
    "countries"   -> country         (list[str] ISO-2; skipped if empty)
    "os"          -> os              (str; skipped if value is "All")
    "formats"     -> format          (list[str]; skipped if empty)
    """
    mask = pd.Series(True, index=df.index)

    advertiser = filters.get("advertiser")
    if advertiser is not None and "advertiser_name" in df.columns:
        mask &= df["advertiser_name"] == advertiser
        logger.debug("Filter advertiser='{}': {} rows", advertiser, mask.sum())

    campaigns = filters.get("campaigns")
    if campaigns is not None and len(campaigns) > 0 and "campaign_id" in df.columns:
        mask &= df["campaign_id"].isin(campaigns)
        logger.debug("Filter campaigns={}: {} rows", campaigns, mask.sum())

    creatives = filters.get("creatives")
    if creatives is not None and len(creatives) > 0 and "creative_id" in df.columns:
        mask &= df["creative_id"].isin(creatives)
        logger.debug("Filter creatives={}: {} rows", creatives, mask.sum())

    countries = filters.get("countries")
    if countries is not None and len(countries) > 0 and "country" in df.columns:
        mask &= df["country"].isin(countries)
        logger.debug("Filter countries={}: {} rows", countries, mask.sum())

    os_filter = filters.get("os")
    if os_filter is not None and os_filter != "All" and "os" in df.columns:
        mask &= df["os"] == os_filter
        logger.debug("Filter os='{}': {} rows", os_filter, mask.sum())

    formats = filters.get("formats")
    if formats is not None and len(formats) > 0 and "format" in df.columns:
        mask &= df["format"].isin(formats)
        logger.debug("Filter formats={}: {} rows", formats, mask.sum())

    result = df.loc[mask].copy()
    active_filters = sum(
        1
        for k, v in filters.items()
        if v is not None
        and not (k == "os" and v == "All")
        and not (isinstance(v, list) and len(v) == 0)
    )
    logger.info(
        "apply_filters: {} → {} rows ({} filters active)",
        len(df),
        len(result),
        active_filters,
    )
    return result
