from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.fitness_scorer import (
    LinearFitnessScorer,
    SharpeCorrelationScorer,
    TopKFitnessScorer,
)


def _make_corr(char, value, attr, level, corr, p, n) -> dict:
    return {
        "user_characteristic": char,
        "user_characteristic_value": value,
        "creative_attribute": attr,
        "creative_attribute_level": level,
        "correlation": corr,
        "p_value": p,
        "method": "statistical",
        "target_metric": "perf_score",
        "n_creatives": n,
    }


@pytest.fixture
def two_segment_corr() -> pd.DataFrame:
    """Two country segments for novelty_score with different n_creatives."""
    return pd.DataFrame(
        [
            _make_corr("country", "US", "novelty_score", None, 0.4, 0.01, 100),
            _make_corr("country", "CA", "novelty_score", None, 0.2, 0.02, 50),
        ]
    )


def test_build_weighted_profile_weighted_avg(two_segment_corr):
    scorer = LinearFitnessScorer()
    profile = scorer._build_weighted_profile(
        correlations=two_segment_corr,
        target_segments=[("country", "US"), ("country", "CA")],
        method="statistical",
        target_metric="perf_score",
        significance_threshold=0.05,
    )
    # weighted avg = (0.4×100 + 0.2×50) / (100+50) = 50/150 ≈ 0.3333
    assert len(profile) == 1
    assert profile.iloc[0]["creative_attribute"] == "novelty_score"
    assert np.isclose(profile.iloc[0]["weighted_correlation"], 50 / 150)


@pytest.fixture
def simple_profile() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "creative_attribute": ["novelty_score", "has_ugc_style", "theme"],
            "creative_attribute_level": [None, None, "gameplay"],
            "weighted_correlation": [0.3, -0.1, 0.2],
        }
    )


def test_contribution_numeric(simple_profile):
    scorer = LinearFitnessScorer()
    creative = pd.Series(
        {"creative_id": 1, "novelty_score": 0.8, "has_ugc_style": 0.0, "theme": "other"}
    )
    cv = scorer._build_contribution_vector(creative, simple_profile)
    # numeric: 0.8 × 0.3 = 0.24
    assert np.isclose(cv["novelty_score"], 0.24)


def test_contribution_binary(simple_profile):
    scorer = LinearFitnessScorer()
    creative = pd.Series(
        {"creative_id": 1, "novelty_score": 0.0, "has_ugc_style": 1.0, "theme": "other"}
    )
    cv = scorer._build_contribution_vector(creative, simple_profile)
    # binary: 1.0 × (−0.1) = −0.1
    assert np.isclose(cv["has_ugc_style"], -0.1)


def test_contribution_categorical_match(simple_profile):
    scorer = LinearFitnessScorer()
    creative = pd.Series(
        {"creative_id": 1, "novelty_score": 0.0, "has_ugc_style": 0.0, "theme": "gameplay"}
    )
    cv = scorer._build_contribution_vector(creative, simple_profile)
    # categorical match: 1.0 × 0.2 = 0.2
    assert np.isclose(cv["theme__gameplay"], 0.2)


def test_contribution_categorical_no_match(simple_profile):
    scorer = LinearFitnessScorer()
    creative = pd.Series(
        {"creative_id": 1, "novelty_score": 0.0, "has_ugc_style": 0.0, "theme": "family"}
    )
    cv = scorer._build_contribution_vector(creative, simple_profile)
    # no match → key absent (not padded with 0)
    assert "theme__gameplay" not in cv.index


def test_contribution_missing_attribute(simple_profile):
    scorer = LinearFitnessScorer()
    # creative has no novelty_score column
    creative = pd.Series({"creative_id": 1, "has_ugc_style": 1.0, "theme": "gameplay"})
    cv = scorer._build_contribution_vector(creative, simple_profile)
    assert "novelty_score" not in cv.index


@pytest.fixture
def mini_summary() -> pd.DataFrame:
    """Three creatives with distinct novelty_score values."""
    return pd.DataFrame(
        {
            "creative_id": [101, 102, 103],
            "novelty_score": [0.9, 0.1, 0.5],
            "has_ugc_style": [0.0, 1.0, 0.0],
            "theme": ["gameplay", "family", "gameplay"],
        }
    )


@pytest.fixture
def single_segment_corr() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _make_corr("country", "US", "novelty_score", None, 0.5, 0.01, 100),
        ]
    )


def test_score_all_returns_series(mini_summary, single_segment_corr):
    scorer = LinearFitnessScorer()
    scores = scorer.score_all(
        creative_summary=mini_summary,
        correlations=single_segment_corr,
        target_segments=[("country", "US")],
    )
    assert isinstance(scores, pd.Series)


def test_score_all_index_is_creative_id(mini_summary, single_segment_corr):
    scorer = LinearFitnessScorer()
    scores = scorer.score_all(mini_summary, single_segment_corr, [("country", "US")])
    assert set(scores.index) == {101, 102, 103}


def test_score_all_sorted_descending(mini_summary, single_segment_corr):
    scorer = LinearFitnessScorer()
    scores = scorer.score_all(mini_summary, single_segment_corr, [("country", "US")])
    assert scores.is_monotonic_decreasing


def test_score_all_top_creative_is_highest_novelty(mini_summary, single_segment_corr):
    # novelty_score × 0.5 → creative 101 (0.9) should win
    scorer = LinearFitnessScorer()
    scores = scorer.score_all(mini_summary, single_segment_corr, [("country", "US")])
    assert scores.idxmax() == 101


def test_score_all_empty_profile_returns_zeros(mini_summary, single_segment_corr):
    scorer = LinearFitnessScorer()
    # significance_threshold=0.0 → no row has p_value < 0.0 → empty profile
    scores = scorer.score_all(
        mini_summary,
        single_segment_corr,
        [("country", "US")],
        significance_threshold=0.0,
    )
    assert (scores == 0.0).all()


def test_sharpe_prefers_consistent_over_spiked():
    """A creative with moderate scores on all attributes beats one that's
    great on one axis but zero on others."""
    profile = pd.DataFrame(
        {
            "creative_attribute": ["novelty_score", "motion_score", "readability_score"],
            "creative_attribute_level": [None, None, None],
            "weighted_correlation": [0.3, 0.3, 0.3],
        }
    )

    # spiked: novelty=1.0, others=0.0 → contributions=[0.3, 0.0, 0.0]
    spiked = pd.Series({"novelty_score": 1.0, "motion_score": 0.0, "readability_score": 0.0})
    # consistent: all=0.6 → contributions=[0.18, 0.18, 0.18]
    consistent = pd.Series({"novelty_score": 0.6, "motion_score": 0.6, "readability_score": 0.6})

    scorer = SharpeCorrelationScorer()

    cv_spiked = scorer._build_contribution_vector(spiked, profile)
    cv_consistent = scorer._build_contribution_vector(consistent, profile)

    assert scorer._score_one(cv_consistent) > scorer._score_one(cv_spiked)


def test_sharpe_epsilon_prevents_division_by_zero():
    scorer = SharpeCorrelationScorer(epsilon=1e-6)
    # all contributions equal → std=0
    cv = pd.Series({"a": 0.3, "b": 0.3, "c": 0.3})
    score = scorer._score_one(cv)
    assert np.isfinite(score)
    assert score > 0


def test_topk_filter_profile_keeps_k_attributes():
    """_filter_profile should keep exactly k unique creative_attribute values."""
    profile = pd.DataFrame(
        {
            "creative_attribute": ["a", "b", "c", "d"],
            "creative_attribute_level": [None, None, None, None],
            "weighted_correlation": [0.5, 0.1, 0.8, 0.3],
        }
    )
    scorer = TopKFitnessScorer(k=2)
    filtered = scorer._filter_profile(profile)
    assert set(filtered["creative_attribute"]) == {"a", "c"}  # top-2 by |weighted_correlation|


def test_topk_filter_keeps_all_levels_of_selected_attribute():
    """When a categorical attribute is in top-K, all its levels are retained."""
    profile = pd.DataFrame(
        {
            "creative_attribute": ["theme", "theme", "novelty_score"],
            "creative_attribute_level": ["gameplay", "family", None],
            "weighted_correlation": [0.5, 0.4, 0.1],
        }
    )
    scorer = TopKFitnessScorer(k=1)
    filtered = scorer._filter_profile(profile)
    # theme is the top attribute (max |wc|=0.5); both levels retained
    assert set(filtered["creative_attribute"]) == {"theme"}
    assert len(filtered) == 2


def test_topk_score_all_top_k_only(mini_summary, single_segment_corr):
    """TopKFitnessScorer(k=1) scores using only the top attribute."""
    scorer = TopKFitnessScorer(k=1)
    scores = scorer.score_all(mini_summary, single_segment_corr, [("country", "US")])
    assert isinstance(scores, pd.Series)
    assert len(scores) == 3
    assert scores.is_monotonic_decreasing


DATA_DIR = Path(__file__).parent.parent.parent / "data"
CORR_PATH = DATA_DIR / "correlations" / "correlations_statistical_perf_score.parquet"


@pytest.mark.skipif(
    not CORR_PATH.exists(),
    reason="Pre-computed correlations not available — run scripts/precompute_correlations.py first",
)
def test_linear_scorer_integration_real_data():
    summary = pd.read_csv(DATA_DIR / "creative_summary.csv")
    correlations = pd.read_parquet(CORR_PATH)

    scorer = LinearFitnessScorer()
    scores = scorer.score_all(
        creative_summary=summary,
        correlations=correlations,
        target_segments=[("country", "US")],
    )

    assert isinstance(scores, pd.Series)
    assert len(scores) == 1080
    assert scores.index.name == "creative_id"
    assert scores.is_monotonic_decreasing
    assert scores.notna().all()
    assert np.isfinite(scores.values).all()


@pytest.mark.skipif(
    not CORR_PATH.exists(),
    reason="Pre-computed correlations not available — run scripts/precompute_correlations.py first",
)
def test_all_scorers_return_valid_series_real_data():
    summary = pd.read_csv(DATA_DIR / "creative_summary.csv")
    correlations = pd.read_parquet(CORR_PATH)
    target = [("country", "US"), ("country", "CA"), ("os", "iOS")]

    scorers = [
        LinearFitnessScorer(),
        SharpeCorrelationScorer(),
        TopKFitnessScorer(k=10),
    ]
    for scorer in scorers:
        scores = scorer.score_all(summary, correlations, target)
        assert len(scores) == 1080, f"{type(scorer).__name__} returned wrong length"
        assert scores.is_monotonic_decreasing, f"{type(scorer).__name__} not sorted"
        assert np.isfinite(scores.values).all(), f"{type(scorer).__name__} has non-finite scores"
