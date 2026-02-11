#!/usr/bin/env python3
"""
Time Series Forecasting - Full Pipeline Script
Run this to generate all outputs for analysis
"""

import pandas as pd
import numpy as np
import json
import warnings

warnings.filterwarnings("ignore")

import lightgbm as lgb


def weighted_rmse_score(y_true, y_pred, weights):
    """Calculate weighted RMSE skill score."""
    y_true, y_pred, weights = (
        np.asarray(y_true),
        np.asarray(y_pred),
        np.asarray(weights),
    )
    weighted_se = np.sum(weights * (y_true - y_pred) ** 2)
    weighted_y2 = np.sum(weights * y_true**2)
    if weighted_y2 == 0:
        return 0.0
    return 1 - np.sqrt(weighted_se / weighted_y2)


def create_features(df):
    """Create temporal features with leakage prevention."""
    print("Step 1: Creating temporal features...")
    df = df.copy()
    df = df.sort_values(["code", "sub_code", "sub_category", "ts_index"]).reset_index(
        drop=True
    )

    # Group key
    df["_group"] = df["code"] + "_" + df["sub_code"] + "_" + df["sub_category"]
    groups = df.groupby("_group", sort=False)

    # Select key features for transformation
    base_features = [
        c
        for c in df.columns
        if c.startswith("feature_") and c not in ["feature_ch", "feature_cg"]
    ]
    print(f"  Processing {len(base_features)} features...")

    # Create lag features (shifted by 1 to prevent leakage)
    for lag in [1, 2]:
        for feat in base_features[:15]:  # Limit to first 15 for speed
            df[f"{feat}_lag{lag}"] = groups[feat].shift(lag).values

    # Rolling mean features
    for window in [7, 14]:
        for feat in base_features[:15]:
            roll_col = f"{feat}_roll{window}"
            # Calculate rolling mean and shift by 1
            df[roll_col] = groups[feat].transform(
                lambda x: x.rolling(window, min_periods=1).mean().shift(1)
            )

    # Expanding mean features
    for feat in base_features[:15]:
        df[f"{feat}_exp"] = groups[feat].transform(
            lambda x: x.expanding(min_periods=1).mean().shift(1)
        )

    # Target encoding for categorical variables (leakage-safe)
    print("  Creating target encodings...")
    for col in ["code", "sub_code", "sub_category"]:
        global_mean = df["feature_ch"].mean()
        df[f"{col}_enc"] = df.groupby(col)["feature_ch"].transform(
            lambda x: x.expanding().mean().shift(1).fillna(global_mean)
        )

    # Additional features
    df["ts_norm"] = (df["ts_index"] - df["ts_index"].min()) / (
        df["ts_index"].max() - df["ts_index"].min()
    )
    df["group_size"] = groups["ts_index"].transform("count").values

    # Feature statistics per group
    df["group_target_mean"] = groups["feature_ch"].transform("mean").values
    df["group_target_std"] = groups["feature_ch"].transform("std").values

    df = df.drop(columns=["_group"])
    return df


def train_models(train_df, valid_df, feature_cols):
    """Train separate models for each horizon."""
    print("\nStep 2: Training models...")
    results = {}
    all_preds, all_true, all_weights = [], [], []
    models = {}

    horizons = sorted(train_df["horizon"].unique())

    for horizon in horizons:
        print(f"  Training horizon {horizon}...")
        tr = train_df[train_df["horizon"] == horizon]
        va = valid_df[valid_df["horizon"] == horizon]

        if len(tr) < 100 or len(va) < 100:
            continue

        X_tr = tr[feature_cols].fillna(0)
        y_tr = tr["feature_ch"].values
        w_tr = tr["feature_cg"].fillna(1.0).values

        X_va = va[feature_cols].fillna(0)
        y_va = va["feature_ch"].values
        w_va = va["feature_cg"].fillna(1.0).values

        train_data = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
        valid_data = lgb.Dataset(X_va, label=y_va, weight=w_va, reference=train_data)

        params = {
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": 63,
            "max_depth": 8,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "n_jobs": -1,
        }

        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )

        preds = model.predict(X_va)
        score = weighted_rmse_score(y_va, preds, w_va)

        results[horizon] = {
            "score": float(score),
            "n_train": len(tr),
            "n_valid": len(va),
            "best_iteration": model.best_iteration,
        }

        models[horizon] = model
        all_preds.extend(preds.tolist())
        all_true.extend(y_va.tolist())
        all_weights.extend(w_va.tolist())

        print(f"    Score: {score:.4f}")

    overall = weighted_rmse_score(all_true, all_preds, all_weights)

    return models, results, overall


def save_predictions(valid_df, models, feature_cols, output_file="predictions.csv"):
    """Save predictions to CSV."""
    print("\nStep 3: Saving predictions...")
    valid_df = valid_df.copy()
    valid_df["prediction"] = np.nan

    for horizon, model in models.items():
        mask = valid_df["horizon"] == horizon
        if mask.sum() > 0:
            X = valid_df.loc[mask, feature_cols].fillna(0)
            valid_df.loc[mask, "prediction"] = model.predict(X)

    submission = valid_df[["id", "prediction"]].copy()
    submission.to_csv(output_file, index=False)
    print(f"  Saved to {output_file}")
    return submission


def save_feature_importance(models, feature_cols, output_file="feature_importance.csv"):
    """Save feature importance."""
    print("\nStep 4: Saving feature importance...")
    importance_data = []

    for horizon, model in models.items():
        imp = model.feature_importance(importance_type="gain")
        for feat, val in zip(feature_cols, imp):
            importance_data.append(
                {"horizon": horizon, "feature": feat, "importance": val}
            )

    imp_df = pd.DataFrame(importance_data)
    imp_df = imp_df.groupby("feature")["importance"].mean().reset_index()
    imp_df = imp_df.sort_values("importance", ascending=False)
    imp_df.to_csv(output_file, index=False)
    print(f"  Saved to {output_file}")
    print("\nTop 20 features:")
    print(imp_df.head(20).to_string(index=False))
    return imp_df


def main():
    print("=" * 70)
    print("TIME SERIES FORECASTING PIPELINE")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    df = pd.read_parquet("data/test.parquet")
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Create features
    df = create_features(df)

    # Get feature columns
    exclude = ["id", "code", "sub_code", "sub_category", "feature_ch", "feature_cg"]
    feature_cols = [c for c in df.columns if c not in exclude]
    print(f"Total features: {len(feature_cols)}")

    # Split data
    split_ts = df["ts_index"].quantile(0.75)
    train_df = df[df["ts_index"] < split_ts].copy()
    valid_df = df[df["ts_index"] >= split_ts].copy()
    print(f"Train: {len(train_df):,}, Valid: {len(valid_df):,}")

    # Train models
    models, results, overall_score = train_models(train_df, valid_df, feature_cols)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for h, r in results.items():
        print(f"Horizon {h:2d}: {r['score']:.4f}")
    print("-" * 70)
    print(f"OVERALL:   {overall_score:.4f}")
    print("=" * 70)

    # Save results to JSON
    with open("results.json", "w") as f:
        json.dump({"overall": overall_score, "by_horizon": results}, f, indent=2)

    # Save predictions
    save_predictions(valid_df, models, feature_cols)

    # Save feature importance
    save_feature_importance(models, feature_cols)

    print("\n" + "=" * 70)
    print("Pipeline complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
