"""
Time Series Forecasting Solution - Iteration D (Polars)
PCA and Target Encoding
"""

import polars as pl
import numpy as np
import lightgbm as lgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

def apply_pca(train_df, valid_df, feature_cols, n_components=10):
    print(f"Applying PCA with {n_components} components...")
    
    X_train = train_df.select(feature_cols).fill_null(0).to_numpy()
    X_valid = valid_df.select(feature_cols).fill_null(0).to_numpy()
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_valid_pca = pca.transform(X_valid_scaled)
    
    pca_cols = [f"pca_{i}" for i in range(n_components)]
    
    # Add back columns
    train_pca = pl.DataFrame(X_train_pca, schema=pca_cols)
    valid_pca = pl.DataFrame(X_valid_pca, schema=pca_cols)
    
    train_df = pl.concat([train_df, train_pca], how="horizontal")
    valid_df = pl.concat([valid_df, valid_pca], how="horizontal")
    
    print(f"Explained Variance: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    return train_df, valid_df, pca_cols

def create_target_encoding(df, col, target="feature_ch", smoothing=10):
    # Sort
    df = df.sort([col, "ts_index"])
    global_mean = df[target].mean()
    
    # Expanding Mean with Smoothing (Shifted)
    return df.with_columns(
        (
            (pl.col(target).shift(1).cum_sum().over(col).fill_null(0) + smoothing * global_mean) 
            / 
            (pl.col(target).shift(1).cum_count().over(col).fill_null(0) + smoothing)
        ).alias(f"{col}_enc")
    )

if __name__ == "__main__":
    print("=" * 60)
    print("Iteration D: PCA & Target Encoding (Polars)")
    print("=" * 60)
    
    # Load
    df = pl.read_parquet("data/test.parquet")
    
    # Split
    split_ts = df["ts_index"].quantile(0.75)
    train_df = df.filter(pl.col("ts_index") < split_ts)
    valid_df = df.filter(pl.col("ts_index") >= split_ts)
    
    # PCA
    num_feats = [c for c in df.columns if c.startswith("feature_")][:50]
    train_df, valid_df, pca_cols = apply_pca(train_df, valid_df, num_feats)
    
    # Encoding
    print("Target Encoding...")
    # Combine to encode (careful with leakage, use shift)
    # Actually, simpler to encode whole DF then re-split
    full_df = pl.concat([train_df, valid_df])
    
    for col in ["code", "sub_code"]:
        full_df = create_target_encoding(full_df, col)
        
    train_df = full_df.filter(pl.col("ts_index") < split_ts)
    valid_df = full_df.filter(pl.col("ts_index") >= split_ts)
    
    print(f"Features added: {pca_cols} + encodings")
    print("Iteration D complete!")
