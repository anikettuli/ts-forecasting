"""
Iteration F: Feature Selection & Interaction Engineering (Polars)
"""

import polars as pl
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

def create_interactions(df, base_feats):
    print("Creating interactions...")
    # Polars expressions
    new_cols = []
    for feat in base_feats:
         # Check if col exists
         if feat in df.columns:
             new_cols.append((pl.col(feat) * pl.col("horizon")).alias(f"{feat}_x_horizon"))
             
    new_cols.append((pl.col("horizon") ** 2).alias("horizon_squared"))
    return df.with_columns(new_cols)

def select_features(train_df, valid_df, feature_cols, k=100):
    print(f"Selecting top {k} features...")
    
    # Needs Numpy
    X_train = train_df.select(feature_cols).fill_null(0).to_numpy()
    y_train = train_df["feature_ch"].to_numpy() # Target
    
    selector = SelectKBest(score_func=f_regression, k=min(k, len(feature_cols)))
    selector.fit(X_train, y_train)
    
    selected_indices = selector.get_support(indices=True)
    selected_feats = [feature_cols[i] for i in selected_indices]
    
    print(f"Selected {len(selected_feats)} features.")
    return selected_feats

if __name__ == "__main__":
    print("Iteration F: Feature Selection (Polars)")
    
    # Load
    df = pl.read_parquet("data/test.parquet")
    
    # Dummy setup for features
    exclude = ["id", "code", "sub_code", "sub_category", "feature_ch", "feature_cg", "ts_index", "horizon"]
    feats = [c for c in df.columns if c not in exclude][:20]
    
    # Interactions
    df = create_interactions(df, feats)
    
    # Split
    split_ts = df["ts_index"].quantile(0.75)
    train_df = df.filter(pl.col("ts_index") < split_ts)
    valid_df = df.filter(pl.col("ts_index") >= split_ts)
    
    # Select
    candidates = feats + [f"{f}_x_horizon" for f in feats] + ["horizon_squared"]
    # Filter candidates existing in df
    candidates = [c for c in candidates if c in df.columns]
    
    selected = select_features(train_df, valid_df, candidates)
    print("Done.")
