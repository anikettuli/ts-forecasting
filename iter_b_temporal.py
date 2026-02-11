"""
Time Series Forecasting Solution - Iteration B (Polars)
Temporal Feature Engineering with Leakage Prevention
"""

import polars as pl
import numpy as np
from typing import Tuple, List
import warnings
import os

warnings.filterwarnings("ignore")


def create_temporal_features(
    df: pl.DataFrame,
    feature_cols: List[str],
    group_cols: List[str] = ["code", "sub_code", "sub_category"],
    rolling_windows: List[int] = [3, 7, 14, 30],
) -> pl.DataFrame:
    """
    Create temporal features with proper leakage prevention using Polars.
    """
    print(f"Creating temporal features for {len(feature_cols)} features...")
    
    # Sort for correct temporal calculations
    df = df.sort(group_cols + ["ts_index"])
    
    # Define expressions list
    exprs = []
    
    # Process subsets if too many features to avoid memory explosion? 
    # Polars is lazy/efficient, but let's be careful.
    # For now, process all passed features.
    
    for feat in feature_cols:
        # Lags
        for lag in [1, 2, 3]:
            exprs.append(pl.col(feat).shift(lag).over(group_cols).alias(f"{feat}_lag_{lag}"))
            
        # Rolling Means (Shifted by 1)
        for window in rolling_windows:
            col_name = f"{feat}_roll_{window}"
            # Shift 1 then roll.
            # Note: rolling_mean is deprecated in some versions, using rolling().mean()
            exprs.append(
                pl.col(feat)
                .shift(1)
                .rolling_mean(window_size=window, min_periods=1)
                .over(group_cols)
                .alias(col_name)
            )
            
        # Expanding Mean (Shifted)
        shifted = pl.col(feat).shift(1)
        exprs.append(
            (shifted.cum_sum() / shifted.cum_count())
            .over(group_cols)
            .alias(f"{feat}_exp_mean")
            .fill_nan(0)
        )
        
    # Apply expressions
    df = df.with_columns(exprs)
    
    print(f"Created {len(exprs)} temporal features")
    return df


def prepare_data_with_features(
    filepath: str = "data/test.parquet",
    target_col: str = "feature_ch",
    weight_col: str = "feature_cg",
    valid_ratio: float = 0.25,
) -> Tuple[pl.DataFrame, pl.DataFrame, List[str]]:
    
    if not os.path.exists(filepath):
        print(f"File {filepath} not found.")
        return pl.DataFrame(), pl.DataFrame(), []

    print("Loading data...")
    df = pl.read_parquet(filepath)
    
    exclude_cols = ["id", "code", "sub_code", "sub_category", target_col, weight_col, "ts_index", "horizon"]
    base_features = [c for c in df.columns if c not in exclude_cols]
    
    # Limit features for speed in demo
    base_features = base_features[:20] if len(base_features) > 50 else base_features
    
    df = create_temporal_features(df, base_features)
    
    # Split
    min_ts = df["ts_index"].min()
    max_ts = df["ts_index"].max()
    split_ts = max_ts - int((max_ts - min_ts) * valid_ratio)
    
    train_df = df.filter(pl.col("ts_index") < split_ts)
    valid_df = df.filter(pl.col("ts_index") >= split_ts)
    
    all_feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    print(f"Train: {train_df.height}, Valid: {valid_df.height}")
    return train_df, valid_df, all_feature_cols


if __name__ == "__main__":
    print("=" * 60)
    print("Iteration B: Temporal Feature Engineering (Polars)")
    print("=" * 60)

    train_df, valid_df, feature_cols = prepare_data_with_features()
    
    if train_df.height > 0:
        temporal_cols = [c for c in train_df.columns if "_roll_" in c or "_lag_" in c]
        print(f"\nSample temporal features ({len(temporal_cols)} total):")
        print(temporal_cols[:5])

    print("\nIteration B complete!")
