"""Train two LightGBM regressors from first-7-day creative signals.

Model 1 — profitability_predictor:
    Target: first day where ROAS has been <= 1 for 3 consecutive days (event_day).

Model 2 — fatigue_predictor:
    Target: fatigue_day from creative_summary (platform-level fatigue label).

Usage:
    uv run python -m src.models.train_fatigue_predictor
"""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.data.loader import load_creative_summary, load_daily_stats
from src.features.early_signals import CATEGORICAL_FEATURES, build_features
from src.features.roas_event import compute_roas_event

_MODEL_DIR = Path(__file__).parent.parent.parent / "data" / "models"


def _log_section(title: str) -> None:
    logger.info("-" * 50)
    logger.info("  {}", title)
    logger.info("-" * 50)


def _train_and_eval(
    features_df: pd.DataFrame,
    y: pd.Series,
    cat_cols: list[str],
    label: str,
) -> tuple[lgb.LGBMRegressor, float]:
    """Train a LGBMRegressor, log evaluation metrics, return (model, mae)."""
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, y, test_size=0.2, random_state=42
    )
    logger.info("  Train: {} | Test: {}", len(X_train), len(X_test))

    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_train,
        y_train,
        categorical_feature=cat_cols,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)],
    )
    logger.info("  Best iteration: {}", model.best_iteration_)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    logger.info("  MAE  : {:.2f} days", mae)
    logger.info("  RMSE : {:.2f} days", rmse)
    logger.info("  R2   : {:.4f}", r2)

    importances = pd.Series(
        model.feature_importances_, index=features_df.columns, name="importance"
    ).sort_values(ascending=False)
    logger.info("  Top-10 features ({}):", label)
    for feat, imp in importances.head(10).items():
        logger.info("    {:<35} {:.0f}", feat, imp)

    return model, mae


def _save(
    model: lgb.LGBMRegressor,
    feature_cols: list[str],
    cat_cols: list[str],
    target: str,
    mae: float,
    stem: str,
) -> None:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = _MODEL_DIR / f"{stem}.pkl"
    meta_path = _MODEL_DIR / f"{stem}_meta.json"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(meta_path, "w") as f:
        json.dump(
            {"features": feature_cols, "categoricals": cat_cols, "target": target, "mae": mae},
            f,
            indent=2,
        )
    logger.info("  Saved: {}", model_path)
    logger.info("  Meta : {}", meta_path)


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    _log_section("Loading data")
    daily_df = load_daily_stats()
    summary_df = load_creative_summary()
    logger.info("daily_df: {} rows | summary_df: {} rows", len(daily_df), len(summary_df))

    # ------------------------------------------------------------------
    # 2. Build shared feature matrix (first 7 days + static attributes)
    # ------------------------------------------------------------------
    _log_section("Engineering features (days 1-7)")
    features = build_features(daily_df, summary_df)
    logger.info(
        "  Feature matrix: {} creatives x {} features", features.shape[0], features.shape[1]
    )

    feature_cols = list(features.columns)
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in features.columns]

    def _prep_x(data: pd.DataFrame) -> pd.DataFrame:
        X = data[feature_cols].copy()
        for col in cat_cols:
            X[col] = X[col].astype("category")
        return X

    # ------------------------------------------------------------------
    # 3. Model 1: Profitability End (ROAS <= 1 for 3 consecutive days)
    # ------------------------------------------------------------------
    _log_section("Model 1: Profitability End (ROAS <= 1 streak of 3)")
    events = compute_roas_event(daily_df, streak=3)
    n_events = events["event_occurred"].sum()
    pct = 100 * n_events / len(events)
    logger.info("  Event count : {} / {} ({:.1f}%)", n_events, len(events), pct)
    logger.info("  Censored    : {}", len(events) - n_events)

    events_idx = events.set_index("creative_id")
    prof_data = features.join(events_idx[["event_occurred", "event_day"]], how="inner")
    prof_data = prof_data[prof_data["event_occurred"]].copy()
    logger.info("  Training samples: {}", len(prof_data))

    X_prof = _prep_x(prof_data)
    y_prof = prof_data["event_day"].astype(float)

    prof_model, prof_mae = _train_and_eval(X_prof, y_prof, cat_cols, "profitability")
    _save(prof_model, feature_cols, cat_cols, "event_day", prof_mae, "profitability_predictor")

    # ------------------------------------------------------------------
    # 4. Model 2: Fatigue Day (platform-level label from creative_summary)
    # ------------------------------------------------------------------
    _log_section("Model 2: Fatigue Day (creative_summary.fatigue_day)")
    fatigued = summary_df[summary_df["fatigue_day"].notna()].copy()
    fatigued = fatigued.set_index("creative_id")
    fat_data = features.join(fatigued[["fatigue_day"]], how="inner").dropna(subset=["fatigue_day"])
    logger.info("  Fatigued creatives: {}", len(fat_data))
    logger.info(
        "  fatigue_day: mean={:.1f}  median={:.1f}  range=[{:.0f}, {:.0f}]",
        fat_data["fatigue_day"].mean(),
        fat_data["fatigue_day"].median(),
        fat_data["fatigue_day"].min(),
        fat_data["fatigue_day"].max(),
    )

    if len(fat_data) < 30:
        logger.warning("  Too few samples — skipping fatigue model.")
    else:
        X_fat = _prep_x(fat_data)
        y_fat = fat_data["fatigue_day"].astype(float)
        fat_model, fat_mae = _train_and_eval(X_fat, y_fat, cat_cols, "fatigue")
        _save(fat_model, feature_cols, cat_cols, "fatigue_day", fat_mae, "fatigue_predictor")


if __name__ == "__main__":
    main()
