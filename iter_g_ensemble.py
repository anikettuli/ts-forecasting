"""
Iteration G: Ensemble & Hyperparameter Tuning (Polars)
"""

import polars as pl
import numpy as np
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

def weighted_rmse_score(y_true, y_pred, weights):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    weights = np.asarray(weights)
    weighted_se = np.sum(weights * (y_true - y_pred) ** 2)
    weighted_y2 = np.sum(weights * y_true**2)
    if weighted_y2 == 0: return 0.0
    return 1 - np.sqrt(weighted_se / weighted_y2)

def train_lgb_model(train_df, valid_df, features, params):
    X_train = train_df.select(features).fill_null(0).to_numpy()
    y_train = train_df["feature_ch"].to_numpy()
    w_train = train_df["feature_cg"].fill_null(1.0).to_numpy()
    
    X_valid = valid_df.select(features).fill_null(0).to_numpy()
    y_valid = valid_df["feature_ch"].to_numpy()
    w_valid = valid_df["feature_cg"].fill_null(1.0).to_numpy()
    
    train_data = lgb.Dataset(X_train, label=y_train, weight=w_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, weight=w_valid, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, valid_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(False)]
    )
    return model

if __name__ == "__main__":
    print("Iteration G: Ensemble (Polars)")
    
    df = pl.read_parquet("data/test.parquet")
    exclude = ["id", "code", "sub_code", "sub_category", "feature_ch", "feature_cg", "ts_index", "horizon"]
    feats = [c for c in df.columns if c not in exclude][:20]
    
    split_ts = df["ts_index"].quantile(0.75)
    train_df = df.filter(pl.col("ts_index") < split_ts)
    valid_df = df.filter(pl.col("ts_index") >= split_ts)
    
    param_sets = [
        {"num_leaves": 31, "learning_rate": 0.05, "device": "gpu"},
        {"num_leaves": 63, "learning_rate": 0.03, "device": "gpu"},
    ]
    
    horizons = sorted(train_df["horizon"].unique().to_list())
    
    for h in horizons:
        print(f"Ensemble for Horizon {h}...")
        t_h = train_df.filter(pl.col("horizon") == h)
        v_h = valid_df.filter(pl.col("horizon") == h)
        
        if t_h.height == 0: continue
            
        preds_list = []
        for p in param_sets:
            full_params = {"objective": "regression", "metric": "rmse", "verbose": -1, **p}
            model = train_lgb_model(t_h, v_h, feats, full_params)
            preds = model.predict(v_h.select(feats).fill_null(0).to_numpy())
            preds_list.append(preds)
            
        avg_preds = np.mean(preds_list, axis=0)
        score = weighted_rmse_score(v_h["feature_ch"].to_numpy(), avg_preds, v_h["feature_cg"].fill_null(1.0).to_numpy())
        print(f"  Ensemble Score: {score:.4f}")
        
    print("Done.")
