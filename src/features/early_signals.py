"""Feature engineering from the first 7 days of a creative's daily data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import linregress

_STATIC_NUMERIC = [
    "novelty_score",
    "motion_score",
    "clutter_score",
    "text_density",
    "readability_score",
    "brand_visibility_score",
    "faces_count",
    "product_count",
    "duration_sec",
    "first_7d_ctr",
    "first_7d_cvr",
    "peak_rolling_ctr_5",
    "first_7d_impressions",
]

_STATIC_BINARY = [
    "has_price",
    "has_discount_badge",
    "has_gameplay",
    "has_ugc_style",
]

CATEGORICAL_FEATURES = [
    "format",
    "theme",
    "emotional_tone",
    "language",
    "hook_type",
    "vertical",
]


def _slope(values: pd.Series) -> float:
    """Linear regression slope; returns 0.0 if fewer than 2 points."""
    vals = values.dropna().values
    if len(vals) < 2:
        return 0.0
    return float(linregress(np.arange(len(vals)), vals).slope)


def build_features(daily_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    """Build the feature matrix: one row per creative.

    Parameters
    ----------
    daily_df:
        Raw daily stats from ``load_daily_stats()``.
    summary_df:
        Creative summary from ``load_creative_summary()``.

    Returns
    -------
    DataFrame indexed by creative_id with numeric, binary, and categorical features.
    """
    # --- Early-signal features from days 1–7 ---
    early = daily_df[daily_df["days_since_launch"] <= 7].copy()

    # Aggregate across country/OS per (creative_id, day)
    agg = early.groupby(["creative_id", "days_since_launch"], as_index=False).agg(
        revenue=("revenue_usd", "sum"),
        spend=("spend_usd", "sum"),
        impressions=("impressions", "sum"),
        clicks=("clicks", "sum"),
        conversions=("conversions", "sum"),
    )
    agg["daily_roas"] = agg["revenue"] / agg["spend"].replace(0, float("nan"))
    agg["daily_ctr"] = agg["clicks"] / agg["impressions"].replace(0, float("nan"))
    agg["daily_cvr"] = agg["conversions"] / agg["clicks"].replace(0, float("nan"))

    records = []
    for cid, grp in agg.groupby("creative_id"):
        grp = grp.sort_values("days_since_launch")
        roas = grp["daily_roas"]
        ctr = grp["daily_ctr"]
        cvr = grp["daily_cvr"]

        mean_roas = float(roas.mean()) if roas.notna().any() else float("nan")
        peak_ctr = float(ctr.max()) if ctr.notna().any() else float("nan")
        mean_ctr = float(ctr.mean()) if ctr.notna().any() else float("nan")

        records.append(
            {
                "creative_id": cid,
                # ROAS signals
                "mean_roas_1_7": mean_roas,
                "min_roas_1_7": float(roas.min()) if roas.notna().any() else float("nan"),
                "roas_slope": _slope(roas),
                "roas_day1": float(roas.iloc[0]) if not roas.empty else float("nan"),
                "roas_day7": float(roas.iloc[-1]) if not roas.empty else float("nan"),
                "roas_day1_vs_day7": (
                    float(roas.iloc[0] / roas.iloc[-1])
                    if not roas.empty and roas.iloc[-1] not in (0, float("nan"))
                    else float("nan")
                ),
                "n_days_roas_below_1": int((roas <= 1).sum()),
                # CTR signals
                "mean_ctr_1_7": mean_ctr,
                "ctr_slope": _slope(ctr),
                "peak_to_mean_ctr": (
                    float(peak_ctr / mean_ctr)
                    if mean_ctr and not np.isnan(mean_ctr) and mean_ctr != 0
                    else float("nan")
                ),
                # CVR / volume
                "mean_cvr_1_7": float(cvr.mean()) if cvr.notna().any() else float("nan"),
                "total_spend_1_7": float(grp["spend"].sum()),
                "total_impressions_1_7": float(grp["impressions"].sum()),
            }
        )

    early_features = pd.DataFrame(records).set_index("creative_id")
    logger.info("Early-signal features: {} creatives × {} columns", *early_features.shape)

    # --- Static features from creative_summary ---
    static_cols = _STATIC_NUMERIC + _STATIC_BINARY + CATEGORICAL_FEATURES
    present = [c for c in static_cols if c in summary_df.columns]
    static = summary_df.set_index("creative_id")[present]

    # --- Join ---
    features = early_features.join(static, how="left")
    logger.info("Full feature matrix: {} creatives × {} columns", *features.shape)
    return features
