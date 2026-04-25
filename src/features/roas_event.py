"""Compute the 3-consecutive-days ROAS ≤ 1 event label from daily stats."""

from __future__ import annotations

import math

import pandas as pd
from loguru import logger


def compute_roas_event(daily_df: pd.DataFrame, streak: int = 3) -> pd.DataFrame:
    """Return one row per creative with the event label.

    The event is defined as the first day where ROAS has been ≤ 1 for
    `streak` consecutive days (aggregated across all countries and OS).

    Parameters
    ----------
    daily_df:
        Raw daily stats DataFrame from ``load_daily_stats()``.
    streak:
        Number of consecutive days with ROAS ≤ 1 required to trigger the event.

    Returns
    -------
    DataFrame with columns:
        creative_id     — original creative ID
        event_occurred  — True if the streak was found
        event_day       — days_since_launch of the last day in the streak (NaN if censored)
    """
    # Aggregate across country/OS → one row per (creative_id, days_since_launch)
    agg = daily_df.groupby(["creative_id", "days_since_launch"], as_index=False).agg(
        revenue=("revenue_usd", "sum"), spend=("spend_usd", "sum")
    )
    agg["daily_roas"] = agg["revenue"] / agg["spend"].replace(0, float("nan"))

    results = []
    for cid, grp in agg.groupby("creative_id"):
        grp = grp.sort_values("days_since_launch")
        current_streak = 0
        event_day = float("nan")
        for _, row in grp.iterrows():
            roas = row["daily_roas"]
            if math.isnan(roas) or roas > 1.0:
                current_streak = 0
            else:
                current_streak += 1
                if current_streak >= streak:
                    event_day = float(row["days_since_launch"])
                    break
        results.append(
            {
                "creative_id": cid,
                "event_occurred": not math.isnan(event_day),
                "event_day": event_day,
            }
        )

    df = pd.DataFrame(results)
    n_events = df["event_occurred"].sum()
    logger.info(
        "compute_roas_event: {}/{} creatives hit {}-day streak ({}%)",
        n_events,
        len(df),
        streak,
        f"{100 * n_events / len(df):.1f}",
    )
    return df
