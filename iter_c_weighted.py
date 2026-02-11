"""
Time Series Forecasting Solution - Iteration C (Polars)
Weighted LightGBM Model
"""

import polars as pl
import numpy as np
import lightgbm as lgb
import warnings
import os

warnings.filterwarnings("ignore")

def weighted_rmse_score(y_true, y_pred, weights):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    weights = np.asarray(weights)
    weighted_se = np.sum(weights * (y_true - y_pred) ** 2)
    weighted_y2 = np.sum(weights * y_true**2)
    if weighted_y2 == 0: return 0.0
    return 1 - np.sqrt(weighted_se / weighted_y2)

def train_model(train_df, valid_df, feature_cols):
    # Convert to numpy for LightGBM
    X_train = train_df.select(feature_cols).fill_null(0).to_numpy()
    y_train = train_df["feature_ch"].to_numpy()
    w_train = train_df["feature_cg"].fill_null(1.0).to_numpy()
    
    X_valid = valid_df.select(feature_cols).fill_null(0).to_numpy()
    y_valid = valid_df["feature_ch"].to_numpy()
    w_valid = valid_df["feature_cg"].fill_null(1.0).to_numpy()
    
    train_data = lgb.Dataset(X_train, label=y_train, weight=w_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, weight=w_valid, reference=train_data)
    
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbose": -1,
        "n_jobs": -1,
        "device": "gpu"
    }
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, valid_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    return model

if __name__ == "__main__":
    print("=" * 60)
    print("Iteration C: Weighted LightGBM (Polars)")
    print("=" * 60)
    
    # Simplified flow for testing
    if not os.path.exists("data/test.parquet"):
        print("Data not found.")
        exit()

    df = pl.read_parquet("data/test.parquet")
    
    # Features
    exclude = ["id", "code", "sub_code", "sub_category", "feature_ch", "feature_cg", "ts_index", "horizon"]
    feats = [c for c in df.columns if c not in exclude]
    
    # Split
    split_ts = df["ts_index"].quantile(0.75)
    train_df = df.filter(pl.col("ts_index") < split_ts)
    valid_df = df.filter(pl.col("ts_index") >= split_ts)
    
    print(f"Train: {train_df.height}, Valid: {valid_df.height}")
    
    # Train separate models
    horizons = sorted(train_df["horizon"].unique().to_list())
    models = {}
    
    for h in horizons:
        t_h = train_df.filter(pl.col("horizon") == h)
        v_h = valid_df.filter(pl.col("horizon") == h)
        
        if t_h.height == 0: continue
        
        print(f"Training horizon {h}...")
        model = train_model(t_h, v_h, feats)
        models[h] = model
        
        preds = model.predict(v_h.select(feats).fill_null(0).to_numpy())
        score = weighted_rmse_score(v_h["feature_ch"].to_numpy(), preds, v_h["feature_cg"].fill_null(1.0).to_numpy())
        print(f"  Score: {score:.4f}")

    print("\nIteration C complete!")
