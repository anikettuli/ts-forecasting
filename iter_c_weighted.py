"""
Time Series Forecasting Solution - Iteration C
Weighted LightGBM Model with Optimized Features
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
import warnings

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available")


def weighted_rmse_score(
    y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray
) -> float:
    """Calculate weighted RMSE skill score."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    weights = np.asarray(weights)

    weighted_squared_error = np.sum(weights * (y_true - y_pred) ** 2)
    weighted_y_squared = np.sum(weights * y_true**2)

    if weighted_y_squared == 0:
        return 0.0

    score = 1 - np.sqrt(weighted_squared_error / weighted_y_squared)
    return score


def create_optimized_features(
    df: pd.DataFrame, feature_cols: List[str]
) -> pd.DataFrame:
    """Create temporal features efficiently using vectorized operations."""
    df = df.copy()
    df = df.sort_values(["code", "sub_code", "sub_category", "ts_index"])

    # Only process most important features to save time
    # Select features with least missing values
    missing_counts = df[feature_cols].isnull().sum()
    top_features = missing_counts.nsmallest(20).index.tolist()

    print(f"Creating features for top {len(top_features)} features...")

    for feat in top_features:
        # Group-based lag (shift by 1 within each group)
        df[f"{feat}_lag1"] = df.groupby(["code", "sub_code", "sub_category"])[
            feat
        ].shift(1)

        # Rolling mean with window 7 (shifted)
        df[f"{feat}_roll7"] = df.groupby(["code", "sub_code", "sub_category"])[
            feat
        ].transform(lambda x: x.rolling(7, min_periods=1).mean().shift(1))

    return df


def train_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "feature_ch",
    weight_col: str = "feature_cg",
) -> lgb.Booster:
    """Train LightGBM model with sample weights."""

    # Prepare data
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df[target_col]
    w_train = train_df[weight_col].fillna(1.0)

    X_valid = valid_df[feature_cols].fillna(0)
    y_valid = valid_df[target_col]
    w_valid = valid_df[weight_col].fillna(1.0)

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train, weight=w_train)
    valid_data = lgb.Dataset(
        X_valid, label=y_valid, weight=w_valid, reference=train_data
    )

    # Parameters
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_jobs": -1,
    }

    print("Training LightGBM model...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )

    return model


def train_separate_models(
    train_df: pd.DataFrame, valid_df: pd.DataFrame, feature_cols: List[str]
) -> dict:
    """Train separate models for each horizon."""
    models = {}
    horizons = sorted(train_df["horizon"].unique())

    for horizon in horizons:
        print(f"\n--- Training model for horizon {horizon} ---")
        train_h = train_df[train_df["horizon"] == horizon]
        valid_h = valid_df[valid_df["horizon"] == horizon]

        if len(train_h) == 0 or len(valid_h) == 0:
            continue

        model = train_model(train_h, valid_h, feature_cols)
        models[horizon] = model

        # Evaluate
        X_valid = valid_h[feature_cols].fillna(0)
        y_valid = valid_h["feature_ch"].values
        w_valid = valid_h["feature_cg"].fillna(1.0).values

        preds = model.predict(X_valid)
        score = weighted_rmse_score(y_valid, preds, w_valid)
        print(f"Horizon {horizon} - Weighted RMSE Score: {score:.4f}")

    return models


if __name__ == "__main__":
    print("=" * 60)
    print("Iteration C: Weighted LightGBM Model")
    print("=" * 60)

    # Load data
    print("Loading data...")
    df = pd.read_parquet("data/test.parquet")

    # Get base features
    exclude = ["id", "code", "sub_code", "sub_category", "feature_ch", "feature_cg"]
    base_features = [c for c in df.columns if c not in exclude]

    # Create features
    df = create_optimized_features(df, base_features)

    # Get all feature columns
    all_features = [
        c for c in df.columns if c not in exclude and c != "horizon" and c != "ts_index"
    ]
    all_features.extend(["horizon", "ts_index"])

    # Split
    split_ts = df["ts_index"].quantile(0.75)
    train_df = df[df["ts_index"] < split_ts].copy()
    valid_df = df[df["ts_index"] >= split_ts].copy()

    print(f"\nTrain: {len(train_df):,}, Valid: {len(valid_df):,}")
    print(f"Features: {len(all_features)}")

    # Train separate models per horizon
    models = train_separate_models(train_df, valid_df, all_features)

    # Overall evaluation
    print("\n=== Overall Evaluation ===")
    all_preds = []
    all_true = []
    all_weights = []

    for horizon, model in models.items():
        valid_h = valid_df[valid_df["horizon"] == horizon]
        if len(valid_h) > 0:
            X = valid_h[all_features].fillna(0)
            preds = model.predict(X)
            all_preds.extend(preds)
            all_true.extend(valid_h["feature_ch"].values)
            all_weights.extend(valid_h["feature_cg"].fillna(1.0).values)

    overall_score = weighted_rmse_score(
        np.array(all_true), np.array(all_preds), np.array(all_weights)
    )
    print(f"Overall Weighted RMSE Score: {overall_score:.4f}")

    # Save predictions
    valid_df["prediction"] = np.nan
    for horizon, model in models.items():
        mask = valid_df["horizon"] == horizon
        if mask.sum() > 0:
            X = valid_df.loc[mask, all_features].fillna(0)
            valid_df.loc[mask, "prediction"] = model.predict(X)

    submission = valid_df[["id", "prediction"]].copy()
    submission.to_csv("submission_valid.csv", index=False)
    print(f"\nValidation predictions saved to submission_valid.csv")
