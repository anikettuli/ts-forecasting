"""
Time Series Forecasting Solution - Iteration D
Enhanced Feature Engineering with Target Encoding and PCA
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
import warnings

warnings.filterwarnings("ignore")

import lightgbm as lgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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


def create_target_encoding(
    df: pd.DataFrame, col: str, target: str, weight: str
) -> pd.DataFrame:
    """
    Create target encoding with temporal leakage prevention.
    Uses expanding mean within each group to prevent leakage.
    """
    df = df.copy()
    df = df.sort_values([col, "ts_index"])

    # Calculate expanding mean of target within each group (shifted by 1)
    grouped = df.groupby(col, sort=False)

    # Weighted expanding mean
    def weighted_expanding_mean(x):
        weights = df.loc[x.index, weight].fillna(1.0)
        targets = df.loc[x.index, target]

        result = []
        weighted_sum = 0
        weight_sum = 0

        for i, (t, w) in enumerate(zip(targets, weights)):
            if i > 0:  # Shift by 1
                result.append(
                    weighted_sum / weight_sum if weight_sum > 0 else targets.mean()
                )
            else:
                result.append(targets.mean())  # Global mean for first occurrence
            weighted_sum += t * w
            weight_sum += w

        return pd.Series(result, index=x.index)

    df[f"{col}_target_enc"] = grouped[target].transform(weighted_expanding_mean)

    return df


def create_all_features(
    df: pd.DataFrame, target_col: str = "feature_ch", weight_col: str = "feature_cg"
) -> pd.DataFrame:
    """Create comprehensive feature set."""
    df = df.copy()

    # Sort for proper temporal operations
    df = df.sort_values(["code", "sub_code", "sub_category", "ts_index"]).reset_index(
        drop=True
    )

    # Get feature columns
    exclude = [
        "id",
        "code",
        "sub_code",
        "sub_category",
        target_col,
        weight_col,
        "horizon",
        "ts_index",
    ]
    feature_cols = [
        c
        for c in df.columns
        if c not in exclude and not c.startswith("feature_") == False
    ]
    feature_cols = [c for c in df.columns if c.startswith("feature_")]

    print(f"Processing {len(feature_cols)} base features...")

    # Group key for efficient operations
    df["_group_key"] = df["code"] + "_" + df["sub_code"] + "_" + df["sub_category"]

    # Create features for each base feature
    new_features = []
    for i, feat in enumerate(feature_cols):
        if i % 20 == 0:
            print(f"  Processing feature {i + 1}/{len(feature_cols)}: {feat}")

        # Lag features (1, 2, 3)
        for lag in [1, 2, 3]:
            col_name = f"{feat}_lag{lag}"
            df[col_name] = df.groupby("_group_key")[feat].shift(lag)
            new_features.append(col_name)

        # Rolling statistics
        for window in [7, 14, 30]:
            # Rolling mean (shifted)
            col_name = f"{feat}_rollmean{window}"
            df[col_name] = df.groupby("_group_key")[feat].transform(
                lambda x: x.rolling(window, min_periods=1).mean().shift(1)
            )
            new_features.append(col_name)

            # Rolling std (shifted)
            col_name = f"{feat}_rollstd{window}"
            df[col_name] = df.groupby("_group_key")[feat].transform(
                lambda x: x.rolling(window, min_periods=1).std().shift(1)
            )
            new_features.append(col_name)

        # Expanding mean (shifted)
        col_name = f"{feat}_expmean"
        df[col_name] = df.groupby("_group_key")[feat].transform(
            lambda x: x.expanding(min_periods=1).mean().shift(1)
        )
        new_features.append(col_name)

    print(f"Created {len(new_features)} temporal features")

    # Target encoding for categorical variables
    print("Creating target encodings...")
    for col in ["code", "sub_code", "sub_category"]:
        df = create_target_encoding(df, col, target_col, weight_col)
        new_features.append(f"{col}_target_enc")

    # Add interaction features
    print("Creating interaction features...")
    df["code_subcode"] = df["code"] + "_" + df["sub_code"]
    df["code_subcat"] = df["code"] + "_" + df["sub_category"]

    # Count features per group
    df["group_size"] = df.groupby("_group_key")["ts_index"].transform("count")
    new_features.append("group_size")

    # Time-based features
    df["ts_normalized"] = (df["ts_index"] - df["ts_index"].min()) / (
        df["ts_index"].max() - df["ts_index"].min()
    )
    new_features.append("ts_normalized")

    # Horizon as categorical
    df["horizon_cat"] = df["horizon"].astype("category")

    df = df.drop(columns=["_group_key"])

    return df, new_features


def apply_pca(
    df: pd.DataFrame, feature_cols: List[str], n_components: int = 20
) -> pd.DataFrame:
    """Apply PCA to reduce dimensionality."""
    print(f"Applying PCA with {n_components} components...")

    # Fill NaN values
    X = df[feature_cols].fillna(0)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Add PCA features
    for i in range(n_components):
        df[f"pca_{i}"] = X_pca[:, i]

    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")

    return df, [f"pca_{i}" for i in range(n_components)]


def train_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "feature_ch",
    weight_col: str = "feature_cg",
) -> lgb.Booster:
    """Train LightGBM model with optimized parameters."""

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df[target_col]
    w_train = train_df[weight_col].fillna(1.0)

    X_valid = valid_df[feature_cols].fillna(0)
    y_valid = valid_df[target_col]
    w_valid = valid_df[weight_col].fillna(1.0)

    train_data = lgb.Dataset(X_train, label=y_train, weight=w_train)
    valid_data = lgb.Dataset(
        X_valid, label=y_valid, weight=w_valid, reference=train_data
    )

    # Enhanced parameters
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "max_depth": 8,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
        "n_jobs": -1,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )

    return model


def train_separate_models(
    train_df: pd.DataFrame, valid_df: pd.DataFrame, feature_cols: List[str]
) -> Dict[int, lgb.Booster]:
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
    print("Iteration D: Enhanced Features + Target Encoding + PCA")
    print("=" * 60)

    # Load data
    print("Loading data...")
    df = pd.read_parquet("data/test.parquet")

    # Create features
    df, temporal_features = create_all_features(df)

    # Get all feature columns
    exclude = [
        "id",
        "code",
        "sub_code",
        "sub_category",
        "feature_ch",
        "feature_cg",
        "horizon_cat",
    ]
    base_features = [
        c for c in df.columns if c not in exclude and c not in ["horizon", "ts_index"]
    ]

    # Add PCA features
    pca_features = [
        c
        for c in df.columns
        if c.startswith("feature_")
        and "_lag" not in c
        and "_roll" not in c
        and "_exp" not in c
    ]
    df, pca_cols = apply_pca(df, pca_features[:50], n_components=15)

    # Combine all features
    all_features = (
        temporal_features
        + pca_cols
        + [
            "horizon",
            "ts_normalized",
            "group_size",
            "code_target_enc",
            "sub_code_target_enc",
            "sub_category_target_enc",
        ]
    )

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
    submission.to_csv("submission_iter_d.csv", index=False)
    print(f"\nValidation predictions saved to submission_iter_d.csv")

    # Feature importance
    print("\n=== Top 20 Important Features ===")
    if models:
        model = list(models.values())[0]
        importance = pd.DataFrame(
            {
                "feature": all_features,
                "importance": model.feature_importance(importance_type="gain"),
            }
        ).sort_values("importance", ascending=False)
        print(importance.head(20).to_string(index=False))
