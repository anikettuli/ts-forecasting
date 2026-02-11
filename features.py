"""
Feature engineering module - Temporal features with leakage prevention
"""

import pandas as pd
import numpy as np


def create_temporal_features(
    df, base_features, group_cols=["code", "sub_code", "sub_category"]
):
    """
    Create temporal features with proper leakage prevention.

    Args:
        df: Input dataframe
        base_features: List of feature columns to transform
        group_cols: Columns to group by

    Returns:
        DataFrame with additional temporal features
    """
    df = df.copy()

    # Create group key
    df["_group_key"] = df[group_cols[0]].astype(str)
    for col in group_cols[1:]:
        df["_group_key"] = df["_group_key"] + "_" + df[col].astype(str)

    # Lag features (shifted by 1, 2, 3)
    for feat in base_features:
        for lag in [1, 2, 3]:
            df[f"{feat}_lag{lag}"] = df.groupby("_group_key")[feat].shift(lag)

    # Rolling means (shifted by 1)
    for feat in base_features:
        for window in [7, 14, 30]:
            df[f"{feat}_roll{window}"] = df.groupby("_group_key")[feat].transform(
                lambda x: x.rolling(window, min_periods=1).mean().shift(1)
            )

    # Expanding mean (shifted by 1)
    for feat in base_features:
        df[f"{feat}_exp"] = df.groupby("_group_key")[feat].transform(
            lambda x: x.expanding(min_periods=1).mean().shift(1)
        )

    df = df.drop(columns=["_group_key"])
    return df


def create_target_encoding(df, col, target_col, weight_col):
    """
    Create target encoding with temporal leakage prevention.
    Uses expanding mean within each group.
    """
    df = df.copy()
    global_mean = df[target_col].mean()

    df[f"{col}_enc"] = df.groupby(col)[target_col].transform(
        lambda x: x.expanding().mean().shift(1).fillna(global_mean)
    )

    return df


def create_all_features(df, target_col="feature_ch", weight_col="feature_cg"):
    """
    Create complete feature set.

    Args:
        df: Input dataframe
        target_col: Target column name
        weight_col: Weight column name

    Returns:
        DataFrame with all features, list of feature column names
    """
    df = df.copy()
    df = df.sort_values(["code", "sub_code", "sub_category", "ts_index"]).reset_index(
        drop=True
    )

    # Get base features
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
    base_features = [
        c for c in df.columns if c not in exclude and c.startswith("feature_")
    ]

    print(f"Creating temporal features for {len(base_features)} base features...")
    df = create_temporal_features(df, base_features)

    # Target encodings
    print("Creating target encodings...")
    for col in ["code", "sub_code", "sub_category"]:
        df = create_target_encoding(df, col, target_col, weight_col)

    # Time features
    df["ts_norm"] = (df["ts_index"] - df["ts_index"].min()) / (
        df["ts_index"].max() - df["ts_index"].min()
    )

    # Group statistics
    df["_group_key"] = (
        df["code"].astype(str)
        + "_"
        + df["sub_code"].astype(str)
        + "_"
        + df["sub_category"].astype(str)
    )
    df["group_size"] = df.groupby("_group_key")["ts_index"].transform("count")
    df["group_target_mean"] = df.groupby("_group_key")[target_col].transform("mean")
    df = df.drop(columns=["_group_key"])

    # Interaction
    df["horizon_x_tsnorm"] = df["horizon"] * df["ts_norm"]

    # Get all feature columns
    exclude = ["id", "code", "sub_code", "sub_category", target_col, weight_col]
    feature_cols = [c for c in df.columns if c not in exclude]

    return df, feature_cols
