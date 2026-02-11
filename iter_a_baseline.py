"""
Time Series Forecasting Solution - Iteration A
Weighted RMSE Score Implementation and Data Loading
"""

import pandas as pd
import numpy as np
from typing import Tuple
import warnings

warnings.filterwarnings("ignore")


def weighted_rmse_score(
    y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray
) -> float:
    """
    Calculate weighted RMSE score (Skill Score).

    SkillScore = 1 - sqrt(sum(w * (y - y_hat)^2) / sum(w * y^2))

    Args:
        y_true: Ground truth target values
        y_pred: Predicted values
        weights: Sample weights

    Returns:
        Skill score (higher is better, 1.0 is perfect)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    weights = np.asarray(weights)

    # Calculate weighted RMSE numerator and denominator
    weighted_squared_error = np.sum(weights * (y_true - y_pred) ** 2)
    weighted_y_squared = np.sum(weights * y_true**2)

    # Avoid division by zero
    if weighted_y_squared == 0:
        return 0.0

    # Calculate skill score
    score = 1 - np.sqrt(weighted_squared_error / weighted_y_squared)

    return score


def load_and_split_data(
    filepath: str = "data/test.parquet",
    target_col: str = "feature_ch",
    weight_col: str = "feature_cg",
    valid_ratio: float = 0.25,
) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Load data and split into train/validation based on ts_index.

    Args:
        filepath: Path to parquet file
        target_col: Column to use as target
        weight_col: Column to use as weight
        valid_ratio: Ratio of data to use for validation (last N% by ts_index)

    Returns:
        train_df, valid_df, feature_columns
    """
    print("Loading data...")
    df = pd.read_parquet(filepath)
    print(f"Loaded {len(df):,} rows with {len(df.columns)} columns")

    # Parse ID components (already present as columns, but verify)
    # id format: code__sub_code__sub_category__horizon__ts_index

    # Get feature columns (exclude id, target, weight, and metadata columns)
    exclude_cols = ["id", "code", "sub_code", "sub_category", target_col, weight_col]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    print(f"Using {len(feature_cols)} features")

    # Determine split point based on ts_index
    min_ts = df["ts_index"].min()
    max_ts = df["ts_index"].max()
    ts_range = max_ts - min_ts
    split_ts = max_ts - int(ts_range * valid_ratio)

    print(f"Time index range: {min_ts} to {max_ts}")
    print(f"Validation split at ts_index >= {split_ts}")

    # Split data
    train_df = df[df["ts_index"] < split_ts].copy()
    valid_df = df[df["ts_index"] >= split_ts].copy()

    print(f"Train set: {len(train_df):,} rows")
    print(f"Validation set: {len(valid_df):,} rows")

    return train_df, valid_df, feature_cols


def create_submission(
    df: pd.DataFrame, predictions: np.ndarray, filename: str = "submission.csv"
):
    """Create submission file with id and prediction columns."""
    submission = pd.DataFrame({"id": df["id"], "prediction": predictions})
    submission.to_csv(filename, index=False)
    print(f"Submission saved to {filename}")
    return submission


if __name__ == "__main__":
    # Test the functions
    print("=" * 60)
    print("Iteration A: Golden Split & Metric")
    print("=" * 60)

    # Load and split data
    train_df, valid_df, feature_cols = load_and_split_data()

    # Show sample of data
    print("\n=== Sample Train Data ===")
    print(train_df[["id", "ts_index", "horizon", "feature_ch", "feature_cg"]].head())

    print("\n=== Sample Validation Data ===")
    print(valid_df[["id", "ts_index", "horizon", "feature_ch", "feature_cg"]].head())

    # Test weighted RMSE with dummy predictions
    y_true = valid_df["feature_ch"].values
    weights = valid_df["feature_cg"].fillna(1.0).values  # Fill missing weights with 1.0
    y_pred = np.ones_like(y_true) * y_true.mean()  # Dummy: predict mean

    score = weighted_rmse_score(y_true, y_pred, weights)
    print(f"\n=== Baseline Score (predict mean) ===")
    print(f"Weighted RMSE Skill Score: {score:.4f}")

    print("\nIteration A complete!")
