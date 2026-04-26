"""Train a LightGBM regressor to predict the day a creative becomes unprofitable.

Target: first day where ROAS has been ≤ 1 for 3 consecutive days.
Features: first-7-day daily signals + static creative attributes.

Usage:
    uv run python src/models/train_fatigue_predictor.py
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


def _print_section(title: str) -> None:
    print(f"\n{'-' * 50}")
    print(f"  {title}")
    print(f"{'-' * 50}")


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    _print_section("Loading data")
    daily_df = load_daily_stats()
    summary_df = load_creative_summary()
    logger.info("daily_df: {} rows | summary_df: {} rows", len(daily_df), len(summary_df))

    # ------------------------------------------------------------------
    # 2. Compute event labels
    # ------------------------------------------------------------------
    _print_section("Computing ROAS event labels (3 consecutive days ROAS <= 1)")
    events = compute_roas_event(daily_df, streak=3)
    n_events = events["event_occurred"].sum()
    pct = 100 * n_events / len(events)
    print(f"  Event count : {n_events} / {len(events)} ({pct:.1f}%)")
    print(f"  Censored    : {len(events) - n_events}")
    if events["event_day"].notna().any():
        print(
            f"  Event day   : mean={events['event_day'].mean():.1f}  "
            f"median={events['event_day'].median():.1f}  "
            f"range=[{events['event_day'].min():.0f}, {events['event_day'].max():.0f}]"
        )

    # ------------------------------------------------------------------
    # 3. Build features
    # ------------------------------------------------------------------
    _print_section("Engineering features (days 1–7)")
    features = build_features(daily_df, summary_df)
    print(f"  Feature matrix: {features.shape[0]} creatives × {features.shape[1]} features")

    # ------------------------------------------------------------------
    # 4. Join labels + features, filter to events only
    # ------------------------------------------------------------------
    events = events.set_index("creative_id")
    data = features.join(events, how="inner")

    # Fallback: if too few events, use fatigue_day from summary instead
    if n_events < 50:
        logger.warning(
            "Only {} events found — falling back to fatigue_day from creative_summary", n_events
        )
        _print_section("FALLBACK: predicting fatigue_day from creative_summary")
        fallback = summary_df[summary_df["fatigue_day"].notna()].copy()
        fallback = fallback.set_index("creative_id")
        data = features.join(fallback[["fatigue_day"]], how="inner").dropna(subset=["fatigue_day"])
        target_col = "fatigue_day"
        print(f"  Using {len(data)} fatigued creatives")
    else:
        data = data[data["event_occurred"]].copy()
        target_col = "event_day"

    print(f"  Training samples: {len(data)}")

    # ------------------------------------------------------------------
    # 5. Prepare X / y
    # ------------------------------------------------------------------
    feature_cols = [
        c
        for c in data.columns
        if c not in ("event_occurred", "event_day", "fatigue_day", target_col)
    ]
    X = data[feature_cols].copy()
    y = data[target_col].astype(float)

    # Encode categoricals as LightGBM category dtype
    cat_cols_present = [c for c in CATEGORICAL_FEATURES if c in X.columns]
    for col in cat_cols_present:
        X[col] = X[col].astype("category")

    # ------------------------------------------------------------------
    # 6. Train / test split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    # ------------------------------------------------------------------
    # 7. Train LightGBM
    # ------------------------------------------------------------------
    _print_section("Training LGBMRegressor")
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
        categorical_feature=cat_cols_present,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)],
    )
    print(f"  Best iteration: {model.best_iteration_}")

    # ------------------------------------------------------------------
    # 8. Evaluate
    # ------------------------------------------------------------------
    _print_section("Evaluation on test set")
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"  MAE  : {mae:.2f} days")
    print(f"  RMSE : {rmse:.2f} days")
    print(f"  R²   : {r2:.4f}")

    # ------------------------------------------------------------------
    # 9. Feature importances
    # ------------------------------------------------------------------
    _print_section("Top-15 feature importances (gain)")
    importances = pd.Series(
        model.feature_importances_, index=feature_cols, name="importance"
    ).sort_values(ascending=False)
    for feat, imp in importances.head(15).items():
        print(f"  {feat:<35} {imp:.0f}")

    # ------------------------------------------------------------------
    # 10. Save model + feature names
    # ------------------------------------------------------------------
    _print_section("Saving model")
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = _MODEL_DIR / "fatigue_predictor.pkl"
    features_path = _MODEL_DIR / "feature_names.json"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(features_path, "w") as f:
        json.dump(
            {"features": feature_cols, "categoricals": cat_cols_present, "target": target_col},
            f,
            indent=2,
        )

    print(f"  Model    : {model_path}")
    print(f"  Features : {features_path}")
    print()


if __name__ == "__main__":
    main()
