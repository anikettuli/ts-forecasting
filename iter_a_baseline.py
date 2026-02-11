"""
Time Series Forecasting Solution - Iteration A (Polars)
Weighted RMSE Score Implementation and Data Loading
"""

import polars as pl
import numpy as np
from typing import Tuple, List
import warnings
import os

warnings.filterwarnings("ignore")


def weighted_rmse_score(
    y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray
) -> float:
    """
    Calculate weighted RMSE score (Skill Score).

    SkillScore = 1 - sqrt(sum(w * (y - y_hat)^2) / sum(w * y^2))
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
) -> Tuple[pl.DataFrame, pl.DataFrame, List[str]]:
    """
    Load data and split into train/validation based on ts_index using Polars.
    """
    print("Loading data...")
    if not os.path.exists(filepath):
         print(f"File {filepath} not found. Creating dummy data.")
         df = pl.DataFrame({"ts_index": range(100), "feature_ch": np.random.randn(100), "feature_cg": np.random.rand(100)})
    else:
        df = pl.read_parquet(filepath)
    
    print(f"Loaded {df.height:,} rows with {len(df.columns)} columns")

    # Determine split point based on ts_index
    min_ts = df["ts_index"].min()
    max_ts = df["ts_index"].max()
    ts_range = max_ts - min_ts
    split_ts = max_ts - int(ts_range * valid_ratio)

    print(f"Time index range: {min_ts} to {max_ts}")
    print(f"Validation split at ts_index >= {split_ts}")

    # Split data
    train_df = df.filter(pl.col("ts_index") < split_ts)
    valid_df = df.filter(pl.col("ts_index") >= split_ts)

    print(f"Train set: {train_df.height:,} rows")
    print(f"Validation set: {valid_df.height:,} rows")
    
    exclude_cols = ["id", "code", "sub_code", "sub_category", target_col, weight_col, "horizon", "ts_index"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    return train_df, valid_df, feature_cols


if __name__ == "__main__":
    print("=" * 60)
    print("Iteration A: Golden Split & Metric (Polars)")
    print("=" * 60)

    # Load and split data
    train_df, valid_df, feature_cols = load_and_split_data()

    # Test weighted RMSE with dummy predictions
    y_true = valid_df["feature_ch"].to_numpy()
    weights = valid_df["feature_cg"].fill_null(1.0).to_numpy()
    
    # Baseline: predict mean of training target
    mean_val = train_df["feature_ch"].mean()
    y_pred = np.ones_like(y_true) * mean_val

    score = weighted_rmse_score(y_true, y_pred, weights)
    print(f"\n=== Baseline Score (predict mean: {mean_val:.4f}) ===")
    print(f"Weighted RMSE Skill Score: {score:.4f}")

    print("\nIteration A complete!")
