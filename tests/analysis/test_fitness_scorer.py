import numpy as np
import pandas as pd
import pytest

from src.analysis.fitness_scorer import LinearFitnessScorer


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
