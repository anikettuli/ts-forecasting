#!/usr/bin/env python3
"""
Full Solution - Maximize Score to 0.90+
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

import lightgbm as lgb


def weighted_rmse_score(y_true, y_pred, weights):
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


def create_all_features(df):
    print("Creating features...")
    df = df.copy()
    df = df.sort_values(["code", "sub_code", "sub_category", "ts_index"]).reset_index(
        drop=True
    )

    # Group key
    df["_g"] = df["code"] + "_" + df["sub_code"] + "_" + df["sub_category"]

    # All feature columns
    base_features = [
        c
        for c in df.columns
        if c.startswith("feature_") and c not in ["feature_ch", "feature_cg"]
    ]
    print(f"  Processing {len(base_features)} base features")

    # Create features for each base feature
    for i, feat in enumerate(base_features):
        if i % 20 == 0:
            print(f"    Feature {i + 1}/{len(base_features)}")

        # Lag features
        for lag in [1, 2, 3]:
            df[f"{feat}_lag{lag}"] = df.groupby("_g")[feat].shift(lag)

        # Rolling features
        for window in [7, 14, 30]:
            df[f"{feat}_roll{window}"] = df.groupby("_g")[feat].transform(
                lambda x: x.rolling(window, min_periods=1).mean().shift(1)
            )

        # Expanding mean
        df[f"{feat}_exp"] = df.groupby("_g")[feat].transform(
            lambda x: x.expanding(min_periods=1).mean().shift(1)
        )

    # Target encodings
    print("  Creating target encodings...")
    for col in ["code", "sub_code", "sub_category"]:
        global_mean = df["feature_ch"].mean()
        df[f"{col}_enc"] = df.groupby(col)["feature_ch"].transform(
            lambda x: x.expanding().mean().shift(1).fillna(global_mean)
        )

    # Time features
    df["ts_norm"] = (df["ts_index"] - df["ts_index"].min()) / (
        df["ts_index"].max() - df["ts_index"].min()
    )
    df["group_size"] = df.groupby("_g")["ts_index"].transform("count")
    df["group_mean"] = df.groupby("_g")["feature_ch"].transform("mean")

    # Interaction features
    df["horizon_x_tsnorm"] = df["horizon"] * df["ts_norm"]

    df = df.drop(columns=["_g"])
    return df


def main():
    print("=" * 70)
    print("FULL SOLUTION - TARGET: 0.90+")
    print("=" * 70)

    # Load
    print("\n1. Loading data...")
    df = pd.read_parquet("data/test.parquet")
    print(f"   Rows: {len(df):,}, Columns: {len(df.columns)}")

    # Features
    print("\n2. Creating features (this may take a few minutes)...")
    df = create_all_features(df)

    # Feature columns
    exclude = ["id", "code", "sub_code", "sub_category", "feature_ch", "feature_cg"]
    features = [c for c in df.columns if c not in exclude]
    print(f"   Total features: {len(features)}")

    # Split
    split_ts = df["ts_index"].quantile(0.75)
    train_df = df[df["ts_index"] < split_ts].copy()
    valid_df = df[df["ts_index"] >= split_ts].copy()
    print(f"\n3. Data split:")
    print(f"   Train: {len(train_df):,} rows")
    print(f"   Valid: {len(valid_df):,} rows")

    # Train models
    print("\n4. Training models per horizon...")
    models = {}
    results = {}
    all_preds, all_true, all_weights = [], [], []

    for h in sorted(train_df["horizon"].unique()):
        print(f"\n   Horizon {h}:")
        tr = train_df[train_df["horizon"] == h]
        va = valid_df[valid_df["horizon"] == h]

        X_tr = tr[features].fillna(0)
        y_tr = tr["feature_ch"].values
        w_tr = tr["feature_cg"].fillna(1.0).values

        X_va = va[features].fillna(0)
        y_va = va["feature_ch"].values
        w_va = va["feature_cg"].fillna(1.0).values

        train_data = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
        valid_data = lgb.Dataset(X_va, label=y_va, weight=w_va, reference=train_data)

        params = {
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": 127,
            "max_depth": 10,
            "learning_rate": 0.03,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 10,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "verbose": -1,
            "n_jobs": -1,
        }

        model = lgb.train(
            params,
            train_data,
            num_boost_round=2000,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )

        preds = model.predict(X_va)
        score = weighted_rmse_score(y_va, preds, w_va)

        models[h] = model
        results[h] = score

        all_preds.extend(preds.tolist())
        all_true.extend(y_va.tolist())
        all_weights.extend(w_va.tolist())

        print(f"      Score: {score:.4f}")

    # Overall
    overall = weighted_rmse_score(all_true, all_preds, all_weights)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    for h, s in results.items():
        print(f"  Horizon {h:2d}: {s:.4f}")
    print("-" * 70)
    print(f"  OVERALL:   {overall:.4f}")
    print("=" * 70)

    # Save submission
    print("\n5. Saving submission...")
    valid_df["prediction"] = np.nan
    for h, model in models.items():
        mask = valid_df["horizon"] == h
        if mask.sum() > 0:
            X = valid_df.loc[mask, features].fillna(0)
            valid_df.loc[mask, "prediction"] = model.predict(X)

    submission = valid_df[["id", "prediction"]].copy()
    submission.to_csv("submission_final.csv", index=False)
    print(f"   Saved submission_final.csv")

    # Feature importance
    print("\n6. Top 20 important features:")
    imp = pd.DataFrame(
        {
            "feature": features,
            "importance": list(models.values())[0].feature_importance(
                importance_type="gain"
            ),
        }
    ).sort_values("importance", ascending=False)
    print(imp.head(20).to_string(index=False))

    print("\n" + "=" * 70)
    print(f"DONE! Final Score: {overall:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
