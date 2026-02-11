"""
Optimized Time Series Forecasting - Fast Feature Engineering
"""

import pandas as pd
import numpy as np
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


def fast_create_features(df):
    """Create features efficiently using vectorized operations."""
    df = df.copy()
    df = df.sort_values(["code", "sub_code", "sub_category", "ts_index"])

    # Key features to transform
    key_features = [
        "feature_a",
        "feature_b",
        "feature_c",
        "feature_d",
        "feature_e",
        "feature_f",
        "feature_g",
        "feature_cf",
        "feature_cg",
        "feature_ch",
    ]

    # Group key
    df["_g"] = df["code"] + "_" + df["sub_code"] + "_" + df["sub_category"]

    print("Creating lag and rolling features...")
    for feat in key_features:
        if feat not in df.columns:
            continue

        # Lag 1
        df[f"{feat}_l1"] = df.groupby("_g")[feat].shift(1)

        # Rolling mean 7 (fast computation)
        roll_sum = (
            df.groupby("_g")[feat]
            .rolling(7, min_periods=1)
            .sum()
            .reset_index(0, drop=True)
        )
        df[f"{feat}_r7"] = (
            df.groupby("_g")[feat]
            .shift(1)
            .groupby(df["_g"])
            .transform(lambda x: x.rolling(7, min_periods=1).mean())
        )

    # Target encoding - simple expanding mean per group
    print("Creating target encodings...")
    for col in ["code", "sub_code"]:
        # Global mean for first occurrence, then expanding
        global_mean = df["feature_ch"].mean()
        df[f"{col}_enc"] = df.groupby(col)["feature_ch"].transform(
            lambda x: x.expanding().mean().shift(1).fillna(global_mean)
        )

    # Time features
    df["ts_norm"] = (df["ts_index"] - df["ts_index"].min()) / (
        df["ts_index"].max() - df["ts_index"].min()
    )

    # Group statistics
    df["g_count"] = df.groupby("_g")["ts_index"].transform("count")
    df["g_mean_target"] = df.groupby("_g")["feature_ch"].transform("mean")

    df = df.drop(columns=["_g"])
    return df


def train_and_evaluate(train_df, valid_df, features):
    """Train model and return score."""
    scores = {}
    all_preds, all_true, all_weights = [], [], []

    for h in sorted(train_df["horizon"].unique()):
        tr = train_df[train_df["horizon"] == h]
        va = valid_df[valid_df["horizon"] == h]

        if len(tr) < 100 or len(va) < 100:
            continue

        X_tr, y_tr, w_tr = (
            tr[features].fillna(0),
            tr["feature_ch"],
            tr["feature_cg"].fillna(1.0),
        )
        X_va, y_va, w_va = (
            va[features].fillna(0),
            va["feature_ch"],
            va["feature_cg"].fillna(1.0),
        )

        train_data = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
        valid_data = lgb.Dataset(X_va, label=y_va, weight=w_va, reference=train_data)

        model = lgb.train(
            {
                "objective": "regression",
                "metric": "rmse",
                "num_leaves": 63,
                "learning_rate": 0.05,
                "verbose": -1,
                "n_jobs": -1,
            },
            train_data,
            num_boost_round=500,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )

        preds = model.predict(X_va)
        score = weighted_rmse_score(y_va, preds, w_va)
        scores[h] = score

        all_preds.extend(preds)
        all_true.extend(y_va)
        all_weights.extend(w_va)

        print(f"  Horizon {h}: {score:.4f}")

    overall = weighted_rmse_score(all_true, all_preds, all_weights)
    return overall, scores, model


if __name__ == "__main__":
    print("=" * 60)
    print("Optimized Model with Fast Feature Engineering")
    print("=" * 60)

    # Load
    print("Loading data...")
    df = pd.read_parquet("data/test.parquet")

    # Features
    df = fast_create_features(df)

    # Feature list
    features = [
        c
        for c in df.columns
        if any(x in c for x in ["_l1", "_r7", "_enc", "ts_norm", "g_"])
    ]
    features += ["horizon", "feature_cf", "feature_cg"]

    # Split
    split_ts = df["ts_index"].quantile(0.75)
    train_df = df[df["ts_index"] < split_ts]
    valid_df = df[df["ts_index"] >= split_ts]

    print(f"\nTrain: {len(train_df):,}, Valid: {len(valid_df):,}")
    print(f"Features: {len(features)}")

    # Train
    print("\nTraining models...")
    overall, scores, model = train_and_evaluate(train_df, valid_df, features)

    print(f"\n=== RESULTS ===")
    for h, s in scores.items():
        print(f"  Horizon {h}: {s:.4f}")
    print(f"\n  OVERALL: {overall:.4f}")

    # Save
    valid_df["prediction"] = np.nan
    for h in scores.keys():
        mask = valid_df["horizon"] == h
        X = valid_df.loc[mask, features].fillna(0)
        # Need to retrain or use cached model - simplify for now

    print("\nDone!")
