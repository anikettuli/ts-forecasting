#!/usr/bin/env python3
"""
Iteration A: The "Golden" Split & Metric
Implements weighted_rmse_score and temporal split
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

import lightgbm as lgb


def weighted_rmse_score(y_true, y_pred, weights):
    """
    Calculate Weighted RMSE Skill Score.
    Formula: 1 - sqrt(sum(w * (y - y_hat)^2) / sum(w * y^2))
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    weights = np.asarray(weights)

    weighted_se = np.sum(weights * (y_true - y_pred) ** 2)
    weighted_y2 = np.sum(weights * y_true**2)

    if weighted_y2 == 0:
        return 0.0

    return 1 - np.sqrt(weighted_se / weighted_y2)


def main():
    print("=" * 70)
    print("ITERATION A: Golden Split & Metric")
    print("=" * 70)

    # Load data
    print("\n1. Loading data...")
    df = pd.read_parquet("data/test.parquet")
    print(f"   Rows: {len(df):,}, Columns: {len(df.columns)}")

    # Identify columns
    print("\n2. Identifying columns...")
    print(f"   Target: feature_ch")
    print(f"   Weight: feature_cg")
    print(f"   ID: id")
    print(f"   Time index: ts_index")
    print(f"   Horizons: {sorted(df['horizon'].unique())}")

    # Golden split: last 25% of ts_index as validation
    print("\n3. Creating temporal split (last 25% for validation)...")
    split_ts = df["ts_index"].quantile(0.75)
    train_df = df[df["ts_index"] < split_ts].copy()
    valid_df = df[df["ts_index"] >= split_ts].copy()
    print(
        f"   Train ts_index range: {train_df['ts_index'].min()} - {train_df['ts_index'].max()}"
    )
    print(
        f"   Valid ts_index range: {valid_df['ts_index'].min()} - {valid_df['ts_index'].max()}"
    )
    print(f"   Train size: {len(train_df):,} ({len(train_df) / len(df) * 100:.1f}%)")
    print(f"   Valid size: {len(valid_df):,} ({len(valid_df) / len(df) * 100:.1f}%)")

    # Basic features (just raw features, no engineering yet)
    exclude = ["id", "code", "sub_code", "sub_category", "feature_ch", "feature_cg"]
    base_features = [
        c for c in df.columns if c not in exclude and c != "horizon" and c != "ts_index"
    ]
    base_features = [c for c in base_features if c.startswith("feature_")]

    print(f"\n4. Using {len(base_features)} raw features (no engineering yet)")

    # Train a simple model as baseline
    print("\n5. Training baseline LightGBM model...")
    print("   (No feature engineering, no horizon-specific models yet)")

    # Train on all data together
    X_train = train_df[base_features].fillna(0)
    y_train = train_df["feature_ch"].values
    w_train = train_df["feature_cg"].fillna(1.0).values

    X_valid = valid_df[base_features].fillna(0)
    y_valid = valid_df["feature_ch"].values
    w_valid = valid_df["feature_cg"].fillna(1.0).values

    train_data = lgb.Dataset(X_train, label=y_train, weight=w_train)
    valid_data = lgb.Dataset(
        X_valid, label=y_valid, weight=w_valid, reference=train_data
    )

    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 31,
        "max_depth": 6,
        "learning_rate": 0.1,
        "verbose": -1,
        "n_jobs": -1,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )

    # Predictions
    preds = model.predict(X_valid)
    score = weighted_rmse_score(y_valid, preds, w_valid)

    # Results
    print("\n" + "=" * 70)
    print("RESULTS - ITERATION A (Baseline)")
    print("=" * 70)
    print(f"Weighted RMSE Score: {score:.4f}")
    print("=" * 70)

    # Save submission
    print("\n6. Saving baseline submission...")
    valid_df["prediction"] = preds
    submission = valid_df[["id", "prediction"]].copy()
    submission.to_csv("submission_iter_a.csv", index=False)
    print(f"   Saved: submission_iter_a.csv")

    print("\n" + "=" * 70)
    print("ITERATION A COMPLETE - Ready for Iteration B")
    print("=" * 70)


if __name__ == "__main__":
    main()
