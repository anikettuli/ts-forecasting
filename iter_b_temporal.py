"""
Time Series Forecasting Solution - Iteration B
Temporal Feature Engineering with Leakage Prevention
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
import warnings

warnings.filterwarnings("ignore")


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


def create_temporal_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    group_cols: List[str] = ["code", "sub_code", "sub_category"],
    rolling_windows: List[int] = [3, 7, 14, 30],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Create temporal features with proper leakage prevention.

    For each group (code, sub_code, sub_category), calculate:
    - Rolling means for specified windows (shifted by 1 to prevent leakage)
    - Expanding means (shifted by 1)
    - Lag features

    Args:
        df: Input dataframe (must be sorted by ts_index within groups)
        feature_cols: List of feature columns to transform
        group_cols: Columns to group by
        rolling_windows: Window sizes for rolling statistics
        verbose: Print progress

    Returns:
        DataFrame with additional temporal features
    """
    df = df.copy()

    # Sort by group columns and ts_index to ensure correct order
    df = df.sort_values(group_cols + ["ts_index"]).reset_index(drop=True)

    new_features = []

    if verbose:
        print(f"Creating temporal features for {len(feature_cols)} features...")
        print(f"Groups: {group_cols}")
        print(f"Rolling windows: {rolling_windows}")

    # Process each feature
    for feat in feature_cols:
        if verbose and len(new_features) % 10 == 0:
            print(f"  Processing {feat}...")

        # Group by the specified columns
        grouped = df.groupby(group_cols, sort=False)[feat]

        # Rolling means (shifted by 1 to prevent leakage)
        for window in rolling_windows:
            col_name = f"{feat}_roll_mean_{window}"
            # rolling().shift(1) ensures we only use past values
            df[col_name] = grouped.transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
            new_features.append(col_name)

        # Expanding mean (shifted by 1)
        col_name = f"{feat}_exp_mean"
        df[col_name] = grouped.transform(
            lambda x: x.expanding(min_periods=1).mean().shift(1)
        )
        new_features.append(col_name)

        # Lag features (shifted by 1, 2, 3)
        for lag in [1, 2, 3]:
            col_name = f"{feat}_lag_{lag}"
            df[col_name] = grouped.shift(lag)
            new_features.append(col_name)

    if verbose:
        print(f"Created {len(new_features)} new temporal features")

    return df


def prepare_data_with_features(
    filepath: str = "data/test.parquet",
    target_col: str = "feature_ch",
    weight_col: str = "feature_cg",
    valid_ratio: float = 0.25,
    use_cache: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Load data and create temporal features.

    Returns:
        train_df, valid_df, all_feature_cols
    """
    cache_path = "data/processed_with_features.parquet"

    if use_cache and os.path.exists(cache_path):
        print("Loading cached processed data...")
        df = pd.read_parquet(cache_path)
    else:
        print("Loading raw data...")
        df = pd.read_parquet(filepath)

        # Get base feature columns
        exclude_cols = [
            "id",
            "code",
            "sub_code",
            "sub_category",
            target_col,
            weight_col,
        ]
        base_feature_cols = [c for c in df.columns if c not in exclude_cols]

        print(
            f"Creating temporal features for {len(base_feature_cols)} base features..."
        )
        df = create_temporal_features(df, base_feature_cols, verbose=True)

        # Save cache
        if use_cache:
            os.makedirs("data", exist_ok=True)
            df.to_parquet(cache_path)
            print(f"Cached processed data to {cache_path}")

    # Get all feature columns (base + temporal)
    exclude_cols = ["id", "code", "sub_code", "sub_category", target_col, weight_col]
    all_feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Split by ts_index
    min_ts = df["ts_index"].min()
    max_ts = df["ts_index"].max()
    split_ts = max_ts - int((max_ts - min_ts) * valid_ratio)

    train_df = df[df["ts_index"] < split_ts].copy()
    valid_df = df[df["ts_index"] >= split_ts].copy()

    print(f"\nTrain set: {len(train_df):,} rows")
    print(f"Validation set: {len(valid_df):,} rows")
    print(f"Total features: {len(all_feature_cols)}")

    return train_df, valid_df, all_feature_cols


import os

if __name__ == "__main__":
    print("=" * 60)
    print("Iteration B: Temporal Feature Engineering")
    print("=" * 60)

    # Prepare data with temporal features
    train_df, valid_df, feature_cols = prepare_data_with_features(use_cache=True)

    # Show sample of new features
    print("\n=== Sample of Temporal Features ===")
    temporal_cols = [
        c for c in train_df.columns if "_roll_" in c or "_exp_" in c or "_lag_" in c
    ]
    print(f"New temporal feature columns (showing first 10):")
    for col in temporal_cols[:10]:
        print(f"  - {col}")

    print("\nIteration B complete!")
    print(f"Total features available: {len(feature_cols)}")
