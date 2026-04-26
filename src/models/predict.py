"""Load trained predictors and generate per-creative predictions for the dashboard."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from loguru import logger

from src.features.early_signals import build_features

_MODEL_DIR = Path(__file__).parent.parent.parent / "data" / "models"


@st.cache_resource
def _load_model(stem: str) -> tuple:
    """Load model + metadata by stem name (e.g. 'profitability_predictor')."""
    model_path = _MODEL_DIR / f"{stem}.pkl"
    meta_path = _MODEL_DIR / f"{stem}_meta.json"
    if not model_path.exists() or not meta_path.exists():
        logger.warning("Model not found: {}", model_path)
        return None, None, None, float("nan")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(meta_path) as f:
        meta = json.load(f)
    logger.info("Loaded {} (MAE={:.1f}d)", stem, meta.get("mae", float("nan")))
    return model, meta["features"], meta["categoricals"], float(meta.get("mae", float("nan")))


@st.cache_data
def _predict(
    stem: str, daily_df: pd.DataFrame, summary_df: pd.DataFrame
) -> tuple[pd.Series, float]:
    """Return (predictions Series indexed by creative_id, mae) for the given model stem."""
    model, feature_cols, cat_cols, mae = _load_model(stem)
    if model is None:
        return pd.Series(dtype=float), float("nan")

    features = build_features(daily_df, summary_df)
    available = [c for c in feature_cols if c in features.columns]
    missing = [c for c in feature_cols if c not in features.columns]
    if missing:
        logger.warning("{}: {} features missing", stem, len(missing))

    X = features[available].copy()
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].astype("category")

    preds = model.predict(X)
    return pd.Series(preds, index=features.index, name=stem), mae


def predict_profitability_end(
    daily_df: pd.DataFrame, summary_df: pd.DataFrame
) -> tuple[pd.Series, float]:
    """Predicted day when ROAS drops <= 1 for 3 consecutive days, plus MAE."""
    return _predict("profitability_predictor", daily_df, summary_df)


def predict_fatigue_day(
    daily_df: pd.DataFrame, summary_df: pd.DataFrame
) -> tuple[pd.Series, float]:
    """Predicted platform fatigue day (from creative_summary.fatigue_day), plus MAE."""
    return _predict("fatigue_predictor", daily_df, summary_df)
