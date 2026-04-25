"""Creative strip component for the Smadex Creative Intelligence dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METRIC_COL = {
    "perf_score": "perf_score",
    "ctr": "overall_ctr",
    "cvr": "overall_cvr",
    "roas": "overall_roas",
}

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

_METADATA_COLS = [
    "creative_id",
    "format",
    "creative_status",
    "perf_score",
    "overall_ctr",
    "overall_cvr",
    "overall_roas",
    "language",
    "theme",
    "emotional_tone",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _asset_path(asset_file: str) -> Path:
    """Return the absolute path for an asset_file value from creative_summary."""
    return _PROJECT_ROOT / "data" / asset_file


def _safe_image(col, path: Path, **kwargs) -> bool:
    """Render an image in *col* if the file exists; return True on success."""
    if path.exists():
        col.image(str(path), **kwargs)
        return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_creative_strip(
    your_summary: pd.DataFrame,
    filters: dict,
) -> None:
    """Render the ranked creative strip for the selected advertiser.

    Parameters
    ----------
    your_summary:
        Filtered ``creative_summary`` rows for the current advertiser (and any
        other active filters already applied by the caller).
    filters:
        Filter state dict returned by ``render_filters()``.  The ``"metric"``
        key controls sorting; valid values: ``"perf_score"``, ``"ctr"``,
        ``"cvr"``, ``"roas"``.
    """
    # Initialise selected_creative session state
    if "selected_creative" not in st.session_state:
        st.session_state["selected_creative"] = None

    # ------------------------------------------------------------------
    # Empty-state guard
    # ------------------------------------------------------------------
    if your_summary.empty:
        st.info("No creatives match the current filters.")
        return

    # ------------------------------------------------------------------
    # Resolve metric column and sort
    # ------------------------------------------------------------------
    metric_key: str = filters.get("metric", "perf_score")
    metric_col: str = METRIC_COL.get(metric_key, "perf_score")

    logger.debug(
        "render_creative_strip: metric='{}' → col='{}', total rows={}",
        metric_key,
        metric_col,
        len(your_summary),
    )

    sorted_df = your_summary.sort_values(metric_col, ascending=False).reset_index(drop=True)
    n = min(12, len(sorted_df))
    top_df = sorted_df.head(n)

    # ------------------------------------------------------------------
    # Section header
    # ------------------------------------------------------------------
    st.subheader(f"Top {n} creatives by {metric_key}")

    # ------------------------------------------------------------------
    # Render one column per creative
    # ------------------------------------------------------------------
    cols = st.columns(n)

    for rank_0, (_, row) in enumerate(top_df.iterrows()):
        rank = rank_0 + 1  # 1-based
        creative_id = row["creative_id"]
        metric_val = row[metric_col]
        asset_file = row.get("asset_file", "")
        img_path = _asset_path(asset_file) if asset_file else Path("/nonexistent")

        col = cols[rank_0]
        with col:
            # Image or fallback
            if img_path.exists():
                st.image(str(img_path), use_container_width=True)
            else:
                st.caption("Image not available")
                logger.warning("Asset not found for creative_id={}: {}", creative_id, img_path)

            # Caption with rank, medal emoji for top 3, and metric value
            if rank == 1:
                caption = f"🥇 #{rank} · {metric_val:.3f}"
            elif rank == 2:
                caption = f"🥈 #{rank} · {metric_val:.3f}"
            elif rank == 3:
                caption = f"🥉 #{rank} · {metric_val:.3f}"
            else:
                caption = f"#{rank} · {metric_val:.3f}"
            st.caption(caption)

            # Select button
            if st.button("Select", key=f"sel_{creative_id}"):
                st.session_state["selected_creative"] = creative_id
                logger.debug("Selected creative_id={}", creative_id)

    # ------------------------------------------------------------------
    # Selected creative detail expander
    # ------------------------------------------------------------------
    selected_id = st.session_state.get("selected_creative")
    if selected_id is not None:
        selected_rows = your_summary[your_summary["creative_id"] == selected_id]

        if selected_rows.empty:
            # Selected creative is no longer in the current filter — clear it
            logger.debug("selected_creative={} not in current filter; clearing", selected_id)
            st.session_state["selected_creative"] = None
        else:
            selected_row = selected_rows.iloc[0]
            asset_file = selected_row.get("asset_file", "")
            img_path = _asset_path(asset_file) if asset_file else Path("/nonexistent")

            with st.expander(f"Selected: Creative {selected_id}", expanded=True):
                left, right = st.columns(2)

                with left:
                    if img_path.exists():
                        st.image(str(img_path), width=200)
                    else:
                        st.caption("Image not available")

                with right:
                    # Show key metadata fields (only those present in the DataFrame)
                    present_cols = [c for c in _METADATA_COLS if c in selected_row.index]
                    meta = {c: selected_row[c] for c in present_cols}
                    for key, val in meta.items():
                        st.markdown(f"**{key}:** {val}")
