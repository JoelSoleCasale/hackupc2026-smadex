"""
Correlation engine: pre-computes correlations between creative attributes and
a performance metric, segmented by user/targeting characteristics.

Configuration is driven by module-level dicts — add new attributes or
characteristics there without touching any computation code.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Configuration — extend here to add new attributes or characteristics
# ---------------------------------------------------------------------------

CREATIVE_ATTRIBUTES: dict[str, Literal["numeric", "binary", "categorical"]] = {
    # numeric / continuous
    "text_density": "numeric",
    "readability_score": "numeric",
    "brand_visibility_score": "numeric",
    "clutter_score": "numeric",
    "novelty_score": "numeric",
    "motion_score": "numeric",
    "faces_count": "numeric",
    "product_count": "numeric",
    "copy_length_chars": "numeric",
    "duration_sec": "numeric",
    # binary flags (0/1) — Pearson == point-biserial, so same formula applies
    "has_price": "binary",
    "has_discount_badge": "binary",
    "has_gameplay": "binary",
    "has_ugc_style": "binary",
    # categorical — one-hot encoded, per-level correlation reported
    "theme": "categorical",
    "hook_type": "categorical",
    "dominant_color": "categorical",
    "emotional_tone": "categorical",
    "language": "categorical",
    "format": "categorical",
}

# source types:
#   "campaign" → filter campaigns.csv on campaign_id to find matching creatives
#   "summary"  → column already present in creative_summary (e.g. after loader merge)
#   "daily"    → filter daily stats, aggregate per creative, recompute metric
USER_CHARACTERISTICS: dict[str, dict] = {
    "target_age_segment": {"source": "campaign", "col": "target_age_segment"},
    "target_os": {"source": "campaign", "col": "target_os"},
    "country": {"source": "daily", "col": "country"},
}

PERFORMANCE_METRICS: list[str] = ["perf_score", "overall_ctr", "overall_cvr", "overall_roas"]

MIN_CREATIVES_THRESHOLD: int = 30

OUTPUT_COLUMNS: list[str] = [
    "user_characteristic",
    "user_characteristic_value",
    "creative_attribute",
    "creative_attribute_level",
    "correlation",
    "p_value",
    "method",
    "target_metric",
    "n_creatives",
]


# ---------------------------------------------------------------------------
# Segment preparation
# ---------------------------------------------------------------------------


def _prepare_campaign_segment(
    summary: pd.DataFrame,
    campaigns: pd.DataFrame,
    characteristic: str,
    value: str,
    col: str,
    target_metric: str,
    min_creatives: int,
) -> pd.DataFrame | None:
    """Filter creative_summary to creatives whose campaign matches col==value.

    For target_os, campaigns with "Both" are included in both Android and iOS
    segments because they run on either platform.
    """
    match_values = [value, "Both"] if col == "target_os" else [value]
    if col in summary.columns:
        mask = summary[col].isin(match_values)
        segment = summary.loc[mask].copy()
    else:
        matching = campaigns.loc[campaigns[col].isin(match_values), "campaign_id"]
        segment = summary.loc[summary["campaign_id"].isin(matching)].copy()

    if len(segment) < min_creatives:
        logger.warning(
            "Segment {}={} has {} creatives (< {}); skipping",
            characteristic,
            value,
            len(segment),
            min_creatives,
        )
        return None

    keep = ["creative_id"] + [c for c in CREATIVE_ATTRIBUTES if c in segment.columns]
    if target_metric in segment.columns:
        keep.append(target_metric)
    else:
        logger.warning(
            "Target metric '{}' not found in summary; skipping {}={}",
            target_metric,
            characteristic,
            value,
        )
        return None

    return segment[keep].dropna(subset=[target_metric])


def _prepare_daily_segment(
    daily: pd.DataFrame,
    summary: pd.DataFrame,
    characteristic: str,
    value: str,
    col: str,
    target_metric: str,
    min_creatives: int,
) -> pd.DataFrame | None:
    """Filter daily stats to col==value, aggregate per creative, attach attributes."""
    filtered = daily.loc[daily[col] == value]
    if filtered.empty:
        logger.warning("No daily rows for {}={}; skipping", characteristic, value)
        return None

    agg = (
        filtered.groupby("creative_id")
        .agg(
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
            conversions=("conversions", "sum"),
            revenue_usd=("revenue_usd", "sum"),
            spend_usd=("spend_usd", "sum"),
        )
        .reset_index()
    )

    if target_metric == "overall_ctr":
        agg[target_metric] = agg["clicks"].div(agg["impressions"].replace(0, float("nan")))
    elif target_metric == "overall_cvr":
        agg[target_metric] = agg["conversions"].div(agg["clicks"].replace(0, float("nan")))
    elif target_metric == "overall_roas":
        agg[target_metric] = agg["revenue_usd"].div(agg["spend_usd"].replace(0, float("nan")))
    elif target_metric == "perf_score":
        # perf_score is a lifetime aggregate; join from summary (not segment-specific)
        logger.debug(
            "perf_score for {}={} is joined from lifetime creative_summary values",
            characteristic,
            value,
        )
        perf_map = summary.set_index("creative_id")["perf_score"]
        agg[target_metric] = agg["creative_id"].map(perf_map)
    else:
        raise ValueError(f"Unknown target_metric: {target_metric}")

    attr_cols = ["creative_id"] + [c for c in CREATIVE_ATTRIBUTES if c in summary.columns]
    result = (
        agg[["creative_id", target_metric]]
        .merge(summary[attr_cols], on="creative_id", how="inner")
        .dropna(subset=[target_metric])
    )

    if len(result) < min_creatives:
        logger.warning(
            "Segment {}={} has {} creatives after aggregation (< {}); skipping",
            characteristic,
            value,
            len(result),
            min_creatives,
        )
        return None

    return result


# ---------------------------------------------------------------------------
# Correlation methods
# ---------------------------------------------------------------------------


def _safe_pearsonr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    import warnings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r, p = pearsonr(x, y)
        return float(r), float(p)
    except Exception:
        return float("nan"), float("nan")


def _correlate_statistical(
    df: pd.DataFrame,
    target_metric: str,
    attributes: dict[str, Literal["numeric", "binary", "categorical"]],
) -> list[dict]:
    records = []
    y = df[target_metric].to_numpy(dtype=float)

    for attr, attr_type in attributes.items():
        if attr not in df.columns:
            continue

        if attr_type in ("numeric", "binary"):
            x = df[attr].fillna(df[attr].median()).to_numpy(dtype=float)
            valid = ~(np.isnan(x) | np.isnan(y))
            if valid.sum() < MIN_CREATIVES_THRESHOLD:
                continue
            corr, p_val = _safe_pearsonr(x[valid], y[valid])
            records.append(
                {
                    "creative_attribute": attr,
                    "creative_attribute_level": None,
                    "correlation": corr,
                    "p_value": p_val,
                    "n_creatives": int(valid.sum()),
                }
            )

        elif attr_type == "categorical":
            dummies = pd.get_dummies(df[attr], prefix=attr, drop_first=False, dtype=float)
            valid_y = ~np.isnan(y)
            for col in dummies.columns:
                level = col[len(attr) + 1 :]
                x = dummies[col].to_numpy(dtype=float)
                valid = valid_y & ~np.isnan(x)
                if valid.sum() < MIN_CREATIVES_THRESHOLD:
                    continue
                corr, p_val = _safe_pearsonr(x[valid], y[valid])
                records.append(
                    {
                        "creative_attribute": attr,
                        "creative_attribute_level": level,
                        "correlation": corr,
                        "p_value": p_val,
                        "n_creatives": int(valid.sum()),
                    }
                )

    return records


def _correlate_rf_signed(
    df: pd.DataFrame,
    target_metric: str,
    attributes: dict[str, Literal["numeric", "binary", "categorical"]],
    n_estimators: int = 200,
    max_depth: int = 8,
    random_state: int = 42,
) -> list[dict]:
    from sklearn.ensemble import RandomForestRegressor

    y = df[target_metric].fillna(df[target_metric].median()).to_numpy(dtype=float)

    numeric_cols = [
        a for a, t in attributes.items() if t in ("numeric", "binary") and a in df.columns
    ]
    X_num = df[numeric_cols].fillna(df[numeric_cols].median()).reset_index(drop=True)

    cat_cols = [a for a, t in attributes.items() if t == "categorical" and a in df.columns]
    dummies_list: list[pd.DataFrame] = []
    dummy_meta: list[tuple[str, str, str]] = []  # (original_attr, level, dummy_col)
    for cat in cat_cols:
        dums = pd.get_dummies(df[cat], prefix=cat, drop_first=False, dtype=float).reset_index(
            drop=True
        )
        dummies_list.append(dums)
        for col in dums.columns:
            dummy_meta.append((cat, col[len(cat) + 1 :], col))

    if dummies_list:
        X = pd.concat([X_num] + dummies_list, axis=1)
    else:
        X = X_num

    if len(X) < MIN_CREATIVES_THRESHOLD:
        return []

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns)

    records = []
    n = len(X)

    for attr in numeric_cols:
        imp = float(importances.get(attr, 0.0))
        corr_sign, p_val = _safe_pearsonr(X_num[attr].to_numpy(dtype=float), y)
        signed_imp = float(np.sign(corr_sign)) * imp
        records.append(
            {
                "creative_attribute": attr,
                "creative_attribute_level": None,
                "correlation": signed_imp,
                "p_value": p_val,
                "n_creatives": n,
            }
        )

    for orig_attr, level, dummy_col in dummy_meta:
        imp = float(importances.get(dummy_col, 0.0))
        x_dummy = X[dummy_col].to_numpy(dtype=float)
        corr_sign, p_val = _safe_pearsonr(x_dummy, y)
        signed_imp = float(np.sign(corr_sign)) * imp
        records.append(
            {
                "creative_attribute": orig_attr,
                "creative_attribute_level": level,
                "correlation": signed_imp,
                "p_value": p_val,
                "n_creatives": n,
            }
        )

    return records


# ---------------------------------------------------------------------------
# Engine class
# ---------------------------------------------------------------------------


class CorrelationEngine:
    """
    Pre-computes correlations between creative attributes and a performance metric
    across all user/targeting characteristic segments.

    Parameters
    ----------
    creative_summary : pd.DataFrame
        Output of load_creative_summary() or equivalent — 1 080 rows with all
        creative KPIs and design attributes, merged with advertiser metadata.
    campaigns : pd.DataFrame
        Raw campaigns.csv loaded with pd.read_csv.
    daily_stats : pd.DataFrame
        Output of load_daily_stats() or equivalent — ~192 k daily rows.
    attribute_config : dict, optional
        Override CREATIVE_ATTRIBUTES to restrict or extend the attribute set.
    characteristic_config : dict, optional
        Override USER_CHARACTERISTICS to restrict or extend segment dimensions.
    min_creatives : int, optional
        Minimum creatives per segment required to compute correlations.
    """

    def __init__(
        self,
        creative_summary: pd.DataFrame,
        campaigns: pd.DataFrame,
        daily_stats: pd.DataFrame,
        attribute_config: dict | None = None,
        characteristic_config: dict | None = None,
        min_creatives: int = MIN_CREATIVES_THRESHOLD,
    ) -> None:
        self._summary = creative_summary
        self._campaigns = campaigns
        self._daily = daily_stats
        self._attributes = attribute_config or CREATIVE_ATTRIBUTES
        self._characteristics = characteristic_config or USER_CHARACTERISTICS
        self._min_creatives = min_creatives

    def compute_all(
        self,
        method: Literal["statistical", "rf_signed"],
        target_metric: str = "perf_score",
    ) -> pd.DataFrame:
        """
        Iterate over all (characteristic, value) pairs and compute correlations.

        Returns a flat DataFrame with OUTPUT_COLUMNS columns.
        """
        if target_metric not in PERFORMANCE_METRICS:
            raise ValueError(f"target_metric must be one of {PERFORMANCE_METRICS}")

        all_records: list[dict] = []
        n_segments = sum(
            len(self._get_values(name, cfg)) for name, cfg in self._characteristics.items()
        )
        logger.info(
            "Starting computation: method={} metric={} total_segments={}",
            method,
            target_metric,
            n_segments,
        )

        for char_name, char_cfg in self._characteristics.items():
            for value in self._get_values(char_name, char_cfg):
                logger.debug("Processing {}={}", char_name, value)
                segment_df = self._build_segment(char_name, char_cfg, value, target_metric)
                if segment_df is None or len(segment_df) < self._min_creatives:
                    continue

                if method == "statistical":
                    records = _correlate_statistical(segment_df, target_metric, self._attributes)
                elif method == "rf_signed":
                    records = _correlate_rf_signed(segment_df, target_metric, self._attributes)
                else:
                    raise ValueError(f"Unknown method: {method}")

                for rec in records:
                    rec["user_characteristic"] = char_name
                    rec["user_characteristic_value"] = value
                    rec["method"] = method
                    rec["target_metric"] = target_metric
                all_records.extend(records)

        if not all_records:
            logger.warning("No records produced — check data paths and min_creatives threshold")
            return pd.DataFrame(columns=OUTPUT_COLUMNS)

        result = pd.DataFrame(all_records)[OUTPUT_COLUMNS]
        logger.info("Done: {} correlation records", len(result))
        return result

    def _get_values(self, char_name: str, char_cfg: dict) -> list[str]:
        """Infer unique values for a characteristic from the data."""
        source = char_cfg["source"]
        col = char_cfg["col"]
        if source == "campaign":
            if col in self._campaigns.columns:
                values = sorted(self._campaigns[col].dropna().unique().tolist())
                # "Both" means the campaign targets Android AND iOS — exclude it as
                # its own segment and instead fold those creatives into each platform.
                if col == "target_os":
                    values = [v for v in values if v != "Both"]
                return values
        elif source == "summary":
            if col in self._summary.columns:
                return sorted(self._summary[col].dropna().unique().tolist())
        elif source == "daily":
            if col in self._daily.columns:
                return sorted(self._daily[col].dropna().unique().tolist())
        logger.warning("Could not infer values for {}; column '{}' not found", char_name, col)
        return []

    def _build_segment(
        self,
        char_name: str,
        char_cfg: dict,
        value: str,
        target_metric: str,
    ) -> pd.DataFrame | None:
        source = char_cfg["source"]
        col = char_cfg["col"]
        if source in ("campaign", "summary"):
            return _prepare_campaign_segment(
                self._summary,
                self._campaigns,
                char_name,
                value,
                col,
                target_metric,
                self._min_creatives,
            )
        else:
            return _prepare_daily_segment(
                self._daily,
                self._summary,
                char_name,
                value,
                col,
                target_metric,
                self._min_creatives,
            )
