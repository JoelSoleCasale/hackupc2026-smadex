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
