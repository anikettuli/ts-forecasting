#!/usr/bin/env python3
"""
Iteration G: Ensemble & Hyperparameter Tuning
Uses ensemble of multiple models and optimizes hyperparameters
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

import lightgbm as lgb


def weighted_rmse_score(y_true, y_pred, weights):
    """Calculate Weighted RMSE Skill Score."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    weights = np.asarray(weights)

    weighted_se = np.sum(weights * (y_true - y_pred) ** 2)
    weighted_y2 = np.sum(weights * y_true**2)

    if weighted_y2 == 0:
        return 0.0

    return 1 - np.sqrt(weighted_se / weighted_y2)


def create_smoothed_target_encoding(df, col, target, min_samples=10, smoothing=10):
    """Create smoothed target encoding."""
    df = df.copy()
    df = df.sort_values([col, "ts_index"])

    global_mean = df[target].mean()

    grouped = df.groupby(col, sort=False)

    def smoothed_encoding(x):
        indices = x.index
        result = []

        for i in range(len(indices)):
            if i == 0:
                result.append(global_mean)
            else:
                hist_idx = indices[:i]
                n = len(hist_idx)
                hist_mean = df.loc[hist_idx, target].mean()

                if n < min_samples:
                    enc = (n * hist_mean + smoothing * global_mean) / (n + smoothing)
                else:
                    enc = (n * hist_mean + smoothing * global_mean) / (n + smoothing)

                result.append(enc)

        return pd.Series(result, index=x.index)

    df[f"{col}_enc_smooth"] = grouped[target].transform(smoothed_encoding)

    return df


def create_enhanced_features(df):
    """Create enhanced features with interactions."""
    df = df.copy()
    df = df.sort_values(["code", "sub_code", "sub_category", "ts_index"]).reset_index(
        drop=True
    )

    df["_group"] = df["code"] + "_" + df["sub_code"] + "_" + df["sub_category"]

    base_features = [
        c
        for c in df.columns
        if c.startswith("feature_") and c not in ["feature_ch", "feature_cg"]
    ]

    print(f"   Processing {len(base_features)} base features...")

    new_features = []

    for i, feat in enumerate(base_features):
        if i % 20 == 0:
            print(f"     Feature {i + 1}/{len(base_features)}")

        for lag in [1, 2, 3]:
            col_name = f"{feat}_lag{lag}"
            df[col_name] = df.groupby("_group")[feat].shift(lag)
            new_features.append(col_name)

        for window in [7, 14, 30]:
            col_name = f"{feat}_rollmean{window}"
            df[col_name] = df.groupby("_group")[feat].transform(
                lambda x: x.rolling(window, min_periods=1).mean().shift(1)
            )
            new_features.append(col_name)

            col_name = f"{feat}_rollstd{window}"
            df[col_name] = df.groupby("_group")[feat].transform(
                lambda x: x.rolling(window, min_periods=1).std().shift(1)
            )
            new_features.append(col_name)

        col_name = f"{feat}_expmean"
        df[col_name] = df.groupby("_group")[feat].transform(
            lambda x: x.expanding(min_periods=1).mean().shift(1)
        )
        new_features.append(col_name)

        col_name = f"{feat}_diff1"
        df[col_name] = df.groupby("_group")[feat].diff(1)
        new_features.append(col_name)

    print(f"   Created {len(new_features)} temporal features")

    print("   Creating interaction features...")
    for i in range(min(5, len(base_features))):
        feat = base_features[i]
        for lag in [1, 2]:
            lag_col = f"{feat}_lag{lag}"
            if lag_col in df.columns:
                interact_col = f"{feat}_lag{lag}_x_horizon"
                df[interact_col] = df[lag_col] * df["horizon"]
                new_features.append(interact_col)

    print("   Creating smoothed target encodings...")
    for col in ["code", "sub_code", "sub_category"]:
        df = create_smoothed_target_encoding(df, col, "feature_ch")
        new_features.append(f"{col}_enc_smooth")

    df["ts_norm"] = (df["ts_index"] - df["ts_index"].min()) / (
        df["ts_index"].max() - df["ts_index"].min()
    )
    new_features.append("ts_norm")

    df["group_size"] = df.groupby("_group")["ts_index"].transform("count")
    new_features.append("group_size")

    df["horizon_x_tsnorm"] = df["horizon"] * df["ts_norm"]
    new_features.append("horizon_x_tsnorm")

    df["horizon_squared"] = df["horizon"] ** 2
    new_features.append("horizon_squared")

    df = df.drop(columns=["_group"])

    return df, new_features


def train_ensemble_models(train_df, valid_df, feature_cols):
    """Train ensemble of models for each horizon."""
    models = {}
    results = {}
    all_preds, all_true, all_weights = [], [], []

    horizons = sorted(train_df["horizon"].unique())

    # Multiple parameter sets for ensemble
    param_sets = [
        {
            "num_leaves": 127,
            "max_depth": 10,
            "learning_rate": 0.03,
            "min_child_samples": 10,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
        },
        {
            "num_leaves": 63,
            "max_depth": 8,
            "learning_rate": 0.05,
            "min_child_samples": 20,
            "reg_alpha": 0.05,
            "reg_lambda": 0.05,
        },
        {
            "num_leaves": 255,
            "max_depth": 12,
            "learning_rate": 0.02,
            "min_child_samples": 5,
            "reg_alpha": 0.2,
            "reg_lambda": 0.2,
        },
    ]

    for horizon in horizons:
        print(f"\n   Training ensemble for horizon {horizon}...")

        train_h = train_df[train_df["horizon"] == horizon]
        valid_h = valid_df[valid_df["horizon"] == horizon]

        if len(train_h) == 0 or len(valid_h) == 0:
            print(f"     Skipping - no data")
            continue

        X_train = train_h[feature_cols].fillna(0)
        y_train = train_h["feature_ch"].values
        w_train = train_h["feature_cg"].fillna(1.0).values

        X_valid = valid_h[feature_cols].fillna(0)
        y_valid = valid_h["feature_ch"].values
        w_valid = valid_h["feature_cg"].fillna(1.0).values

        # Train multiple models
        horizon_models = []
        horizon_preds = []

        for i, param_set in enumerate(param_sets):
            print(f"     Model {i + 1}/{len(param_sets)}...")

            params = {
                "objective": "regression",
                "metric": "rmse",
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": -1,
                "n_jobs": -1,
                **param_set,
            }

            train_data = lgb.Dataset(X_train, label=y_train, weight=w_train)
            valid_data = lgb.Dataset(
                X_valid, label=y_valid, weight=w_valid, reference=train_data
            )

            model = lgb.train(
                params,
                train_data,
                num_boost_round=2000,
                valid_sets=[valid_data],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
            )

            preds = model.predict(X_valid)
            score = weighted_rmse_score(y_valid, preds, w_valid)

            horizon_models.append(model)
            horizon_preds.append(preds)

            print(f"       Score: {score:.4f}")

        # Ensemble predictions (simple average)
        ensemble_preds = np.mean(horizon_preds, axis=0)
        ensemble_score = weighted_rmse_score(y_valid, ensemble_preds, w_valid)

        models[horizon] = horizon_models
        results[horizon] = ensemble_score

        all_preds.extend(ensemble_preds.tolist())
        all_true.extend(y_valid.tolist())
        all_weights.extend(w_valid.tolist())

        print(f"     Ensemble Score: {ensemble_score:.4f}")

    return models, results, all_preds, all_true, all_weights


def main():
    print("=" * 70)
    print("ITERATION G: Ensemble & Hyperparameter Tuning")
    print("=" * 70)

    print("\n1. Loading data...")
    df = pd.read_parquet("data/test.parquet")
    print(f"   Rows: {len(df):,}, Columns: {len(df.columns)}")

    print("\n2. Creating enhanced features...")
    df, feature_cols = create_enhanced_features(df)
    print(f"   Total features: {len(feature_cols)}")

    print("\n3. Creating temporal split...")
    split_ts = df["ts_index"].quantile(0.75)
    train_df = df[df["ts_index"] < split_ts].copy()
    valid_df = df[df["ts_index"] >= split_ts].copy()
    print(f"   Train: {len(train_df):,}, Valid: {len(valid_df):,}")

    print("\n4. Training ensemble models...")
    models, results, all_preds, all_true, all_weights = train_ensemble_models(
        train_df, valid_df, feature_cols
    )

    overall = weighted_rmse_score(all_true, all_preds, all_weights)

    print("\n" + "=" * 70)
    print("RESULTS - ITERATION G")
    print("=" * 70)
    for horizon, score in results.items():
        print(f"  Horizon {horizon:2d}: {score:.4f}")
    print("-" * 70)
    print(f"  OVERALL:       {overall:.4f}")
    print("=" * 70)

    print("\n5. Saving submission...")
    valid_df["prediction"] = np.nan
    for horizon, horizon_models in models.items():
        mask = valid_df["horizon"] == horizon
        if mask.sum() > 0:
            X = valid_df.loc[mask, feature_cols].fillna(0)
            # Ensemble predictions
            preds_list = [model.predict(X) for model in horizon_models]
            valid_df.loc[mask, "prediction"] = np.mean(preds_list, axis=0)

    submission = valid_df[["id", "prediction"]].copy()
    submission.to_csv("submission_iter_g.csv", index=False)
    print(f"   Saved: submission_iter_g.csv")

    print("\n" + "=" * 70)
    print("ITERATION G COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
