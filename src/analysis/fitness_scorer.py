"""
Creative fitness scorer: ranks existing creatives against a weighted
correlation profile for a given target audience.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd
from loguru import logger


class CreativeFitnessScorer(ABC):
    """Score all creatives in creative_summary against a target audience profile.

    Subclasses implement _score_one(contributions) -> float.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_all(
        self,
        creative_summary: pd.DataFrame,
        correlations: pd.DataFrame,
        target_segments: list[tuple[str, str]],
        method: str = "statistical",
        target_metric: str = "perf_score",
        significance_threshold: float = 0.05,
    ) -> pd.Series:
        """Return fitness scores for every creative, sorted descending.

        Parameters
        ----------
        creative_summary : pd.DataFrame
            1 080-row DataFrame from load_creative_summary() with all design
            attribute columns.
        correlations : pd.DataFrame
            Pre-computed correlation parquet loaded into memory (OUTPUT_COLUMNS
            from correlation_engine.py).
        target_segments : list[tuple[str, str]]
            Audience constraints as (characteristic, value) pairs, e.g.
            [("country", "US"), ("country", "CA"), ("os", "iOS")].
            Multiple pairs for the same characteristic are unioned (not
            intersected).
        method : str
            Filter correlations to this method ("statistical" or "rf_signed").
        target_metric : str
            Filter correlations to this metric (e.g. "perf_score").
        significance_threshold : float
            Drop correlation rows with p_value >= this value.

        Returns
        -------
        pd.Series
            Float fitness scores indexed by creative_id, sorted descending.
        """
        profile = self._build_weighted_profile(
            correlations, target_segments, method, target_metric, significance_threshold
        )
        profile = self._filter_profile(profile)

        if profile.empty:
            logger.warning(
                "Empty profile after filtering — returning zero scores for all {} creatives",
                len(creative_summary),
            )
            result = pd.Series(0.0, index=creative_summary["creative_id"].values)
            result.index.name = "creative_id"
            result.name = "fitness_score"
            return result.sort_values(ascending=False)

        scores = creative_summary.apply(
            lambda row: (
                self._score_one(cv)
                if not (cv := self._build_contribution_vector(row, profile)).empty
                else 0.0
            ),
            axis=1,
        )
        scores.index = creative_summary["creative_id"].values
        scores.index.name = "creative_id"
        scores.name = "fitness_score"
        logger.info(
            "{}: scored {} creatives; top creative_id={}",
            type(self).__name__,
            len(scores),
            scores.idxmax(),
        )
        return scores.sort_values(ascending=False)

    # ------------------------------------------------------------------
    # Base-class internals
    # ------------------------------------------------------------------

    def _build_weighted_profile(
        self,
        correlations: pd.DataFrame,
        target_segments: list[tuple[str, str]],
        method: str,
        target_metric: str,
        significance_threshold: float,
    ) -> pd.DataFrame:
        """Filter correlations to the target segments and compute weighted-avg correlation.

        Returns a DataFrame with columns:
            creative_attribute, creative_attribute_level, weighted_correlation
        """
        if not target_segments:
            logger.warning("target_segments is empty; profile will be empty")
            return pd.DataFrame(
                columns=["creative_attribute", "creative_attribute_level", "weighted_correlation"]
            )

        seg_mask = pd.Series(False, index=correlations.index)
        for char, value in target_segments:
            match = (correlations["user_characteristic"] == char) & (
                correlations["user_characteristic_value"] == value
            )
            if not match.any():
                logger.warning(
                    "Segment ({}={}) not found in correlations table; skipping", char, value
                )
            seg_mask |= match

        subset = correlations.loc[
            seg_mask
            & (correlations["method"] == method)
            & (correlations["target_metric"] == target_metric)
            & (correlations["p_value"] < significance_threshold)
        ].copy()

        if subset.empty:
            logger.warning(
                "No correlation rows matched method={} metric={} threshold={} for {} segments",
                method,
                target_metric,
                significance_threshold,
                len(target_segments),
            )
            return pd.DataFrame(
                columns=["creative_attribute", "creative_attribute_level", "weighted_correlation"]
            )

        subset["_w_corr"] = subset["correlation"] * subset["n_creatives"]
        profile = (
            subset.groupby(["creative_attribute", "creative_attribute_level"], dropna=False)
            .agg(weighted_correlation=("_w_corr", "sum"), _total_n=("n_creatives", "sum"))
            .reset_index()
        )
        profile["weighted_correlation"] /= profile["_total_n"]
        profile = profile.drop(columns=["_total_n"])

        logger.debug("_build_weighted_profile: {} attribute rows in profile", len(profile))
        return profile

    def _build_contribution_vector(self, creative: pd.Series, profile: pd.DataFrame) -> pd.Series:
        """Map a creative's attribute values onto the correlation profile.

        For numeric/binary attributes: contribution = attribute_value × weighted_correlation.
        For categorical attributes: contribution = 1.0 × weighted_correlation when the
        creative's value matches the profile level, 0.0 otherwise (excluded from the vector).

        Returns a pd.Series of named contributions. Attributes absent from the profile
        are excluded entirely (not padded with 0).
        """
        contributions: dict[str, float] = {}
        for _, row in profile.iterrows():
            attr = row["creative_attribute"]
            level = row["creative_attribute_level"]
            wc = float(row["weighted_correlation"])

            if attr not in creative.index or pd.isna(creative[attr]):
                continue

            if pd.isna(level):  # numeric or binary
                contributions[attr] = float(creative[attr]) * wc
            else:  # categorical — only include when the creative matches the level
                if str(creative[attr]) == str(level):
                    contributions[f"{attr}__{level}"] = 1.0 * wc

        return pd.Series(contributions, dtype=float)

    def _filter_profile(self, profile: pd.DataFrame) -> pd.DataFrame:
        """Hook for subclasses to narrow the profile before scoring. Default: identity."""
        return profile

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _score_one(self, contributions: pd.Series) -> float:
        """Aggregate a contribution vector into a scalar fitness score."""
        ...


# ---------------------------------------------------------------------------
# Concrete scorers
# ---------------------------------------------------------------------------


class LinearFitnessScorer(CreativeFitnessScorer):
    """Sum of all attribute contributions — maximises total audience alignment."""

    def _score_one(self, contributions: pd.Series) -> float:
        return float(contributions.sum())


class SharpeCorrelationScorer(CreativeFitnessScorer):
    """Mean / std of contributions — rewards consistent alignment across attributes.

    Parameters
    ----------
    epsilon : float
        Small constant added to std to prevent division by zero.
    """

    def __init__(self, epsilon: float = 1e-6) -> None:
        self.epsilon = epsilon

    def _score_one(self, contributions: pd.Series) -> float:
        return float(contributions.mean() / (contributions.std(ddof=0) + self.epsilon))


class TopKFitnessScorer(CreativeFitnessScorer):
    """Sum contributions for only the K most influential audience-signal attributes.

    Selection is by max |weighted_correlation| per attribute in the profile,
    not by each creative's individual contribution magnitude.

    Parameters
    ----------
    k : int
        Number of top attributes to retain.
    """

    def __init__(self, k: int = 10) -> None:
        self.k = k

    def _filter_profile(self, profile: pd.DataFrame) -> pd.DataFrame:
        if profile.empty:
            return profile
        attr_importance = (
            profile.groupby("creative_attribute")["weighted_correlation"]
            .apply(lambda x: x.abs().max())
            .nlargest(self.k)
        )
        return profile[profile["creative_attribute"].isin(attr_importance.index)].copy()

    def _score_one(self, contributions: pd.Series) -> float:
        return float(contributions.sum())
