"""
Offline pre-computation of creative-attribute × performance-metric correlations,
segmented by user/targeting characteristics.

Usage
-----
    uv run python scripts/precompute_correlations.py --method statistical --metric perf_score
    uv run python scripts/precompute_correlations.py --method rf_signed --metric overall_ctr
    uv run python scripts/precompute_correlations.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR / "correlations"

sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402

from src.analysis.correlation_engine import (  # noqa: E402
    PERFORMANCE_METRICS,
    CorrelationEngine,
)


def _setup_logging(verbose: bool) -> None:
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        level=level,
    )
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logger.add(log_dir / "precompute_correlations_{time}.log", rotation="10 MB", level="DEBUG")


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info("Loading data from {}", DATA_DIR)

    summary_path = DATA_DIR / "creative_summary.csv"
    advertisers_path = DATA_DIR / "advertisers.csv"
    campaigns_path = DATA_DIR / "campaigns.csv"
    daily_path = DATA_DIR / "creative_daily_country_os_stats.csv"

    summary = pd.read_csv(summary_path)
    logger.info("creative_summary: {} rows", len(summary))

    try:
        advertisers = pd.read_csv(advertisers_path)
        dup_cols = [
            c for c in advertisers.columns if c != "advertiser_name" and c in summary.columns
        ]
        advertisers = advertisers.drop(columns=dup_cols)
        summary = summary.merge(advertisers, on="advertiser_name", how="left")
        logger.info("creative_summary after advertiser merge: {} rows", len(summary))
    except FileNotFoundError:
        logger.warning("advertisers.csv not found; skipping merge")

    campaigns = pd.read_csv(campaigns_path)
    logger.info("campaigns: {} rows", len(campaigns))

    daily = pd.read_csv(daily_path, parse_dates=["date"])
    logger.info("daily_stats: {} rows", len(daily))

    return summary, campaigns, daily


def _run(
    method: str,
    metric: str,
    min_creatives: int,
    summary: pd.DataFrame,
    campaigns: pd.DataFrame,
    daily: pd.DataFrame,
) -> None:
    engine = CorrelationEngine(
        creative_summary=summary,
        campaigns=campaigns,
        daily_stats=daily,
        min_creatives=min_creatives,
    )
    df = engine.compute_all(method=method, target_metric=metric)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"correlations_{method}_{metric}.parquet"
    df.to_parquet(out_path, index=False)
    logger.success("Saved {} rows → {}", len(df), out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--method",
        choices=["statistical", "rf_signed"],
        default="statistical",
        help="Correlation method (default: statistical)",
    )
    parser.add_argument(
        "--metric",
        choices=PERFORMANCE_METRICS,
        default="perf_score",
        help="Target performance metric (default: perf_score)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="run_all",
        help="Run all method × metric combinations",
    )
    parser.add_argument(
        "--min-creatives",
        type=int,
        default=30,
        dest="min_creatives",
        help="Minimum creatives per segment (default: 30)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    summary, campaigns, daily = _load_data()

    combos = (
        [(m, t) for m in ("statistical", "rf_signed") for t in PERFORMANCE_METRICS]
        if args.run_all
        else [(args.method, args.metric)]
    )

    for method, metric in combos:
        logger.info("--- method={} metric={} ---", method, metric)
        _run(method, metric, args.min_creatives, summary, campaigns, daily)

    logger.success("All done. Output directory: {}", OUT_DIR)


if __name__ == "__main__":
    main()
