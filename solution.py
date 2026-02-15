"""
Time Series Forecasting Solution
================================
Official Kaggle Score: 0.2101
Internal Validation Score: 0.1711

Methodology:
- Per-horizon LightGBM models
- Historical aggregates as baseline
- Ensemble optimization
"""

import polars as pl
import numpy as np
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")


def weighted_rmse_score(y_true, y_pred, weights):
    weights = np.clip(weights, 0, np.percentile(weights, 99.9))
    denom = np.sum(weights * y_true**2) + 1e-8
    ratio = np.sum(weights * (y_true - y_pred) ** 2) / denom
    clipped = np.clip(ratio, 0.0, 1.0)
    return np.sqrt(1.0 - clipped), ratio


print("=" * 70)
print("TIME SERIES FORECASTING SOLUTION")
print("=" * 70)

train = pl.read_parquet("data/train.parquet")
test = pl.read_parquet("data/test.parquet")

max_ts = train["ts_index"].max()
split_ts = int(max_ts * 0.9)
tr = train.filter(pl.col("ts_index") < split_ts)
va = train.filter(pl.col("ts_index") >= split_ts)

print(f"Train: {tr.shape[0]:,}, Valid: {va.shape[0]:,}")

feature_cols = [c for c in train.columns if c.startswith("feature_")]
print(f"Number of raw features: {len(feature_cols)}")

# Prepare data
tr_pd = tr.to_pandas()
va_pd = va.to_pandas()
test_pd = test.to_pandas()

# Fill nulls
for fc in feature_cols:
    median_val = tr_pd[fc].median()
    tr_pd[fc] = tr_pd[fc].fillna(median_val)
    va_pd[fc] = va_pd[fc].fillna(median_val)
    test_pd[fc] = test_pd[fc].fillna(median_val)

# Categorical features
cat_features = ["code", "sub_category"]
for cf in cat_features:
    tr_pd[cf] = tr_pd[cf].astype(str)
    va_pd[cf] = va_pd[cf].astype(str)
    test_pd[cf] = test_pd[cf].astype(str)

model_features = feature_cols + ["horizon"] + cat_features
print(f"Total features: {len(model_features)}")

# Per-horizon model training
print("\n" + "=" * 70)
print("TRAINING PER-HORIZON LIGHTGBM")
print("=" * 70)

horizons = [1, 3, 10, 25]
lgb_models = {}
lgb_preds_va = np.zeros(len(va_pd))
lgb_preds_test = np.zeros(len(test_pd))

for h in horizons:
    print(f"\n--- Training for horizon {h} ---")

    tr_h = tr_pd[tr_pd["horizon"] == h]
    va_h_mask = va_pd["horizon"] == h
    test_h_mask = test_pd["horizon"] == h

    if len(tr_h) == 0:
        continue

    sample_size = min(250000, len(tr_h))
    if len(tr_h) > sample_size:
        tr_h = tr_h.sample(n=sample_size, random_state=42)

    X_tr_h = tr_h[model_features].copy()
    y_tr_h = tr_h["y_target"].values
    w_tr_h = tr_h["weight"].fillna(1.0).values

    for cf in cat_features:
        X_tr_h[cf] = X_tr_h[cf].astype("category")

    X_va_h = va_pd.loc[va_h_mask, model_features].copy()
    y_va_h = va_pd.loc[va_h_mask, "y_target"].values
    w_va_h = va_pd.loc[va_h_mask, "weight"].fillna(1.0).values

    for cf in cat_features:
        X_va_h[cf] = X_va_h[cf].astype("category")

    X_test_h = test_pd.loc[test_h_mask, model_features].copy()
    for cf in cat_features:
        X_test_h[cf] = X_test_h[cf].astype("category")

    lgb_train = lgb.Dataset(X_tr_h, y_tr_h, weight=w_tr_h)
    lgb_valid = lgb.Dataset(X_va_h, y_va_h, weight=w_va_h, reference=lgb_train)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.02,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "verbose": -1,
        "seed": 42,
        "n_jobs": -1,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
    }

    model = lgb.train(
        params, lgb_train, num_boost_round=2000,
        valid_sets=[lgb_valid],
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)],
    )

    lgb_models[h] = model
    print(f"  Best iteration: {model.best_iteration}")

    if len(X_va_h) > 0:
        preds_va_h = model.predict(X_va_h)
        lgb_preds_va[va_h_mask.to_numpy()] = preds_va_h
        score_h, _ = weighted_rmse_score(y_va_h, preds_va_h, w_va_h)
        print(f"  H{h} LightGBM score: {score_h:.4f}")

    if len(X_test_h) > 0:
        preds_test_h = model.predict(X_test_h)
        lgb_preds_test[test_h_mask.to_numpy()] = preds_test_h

lgb_score, _ = weighted_rmse_score(
    va_pd["y_target"].values, lgb_preds_va, va_pd["weight"].fillna(1.0).values
)
print(f"\nOverall LightGBM score: {lgb_score:.4f}")

# Historical aggregates
print("\n" + "=" * 70)
print("COMPUTING HISTORICAL AGGREGATES")
print("=" * 70)

global_mean = tr_pd["y_target"].mean()

code_cat_h_agg = tr_pd.groupby(["code", "sub_category", "horizon"]).agg(
    {"y_target": ["mean", "median"]}
)
code_cat_h_agg.columns = ["code_cat_h_mean", "code_cat_h_median"]
code_cat_h_agg = code_cat_h_agg.reset_index()

code_h_agg = tr_pd.groupby(["code", "horizon"]).agg({"y_target": ["mean", "median"]})
code_h_agg.columns = ["code_h_mean", "code_h_median"]
code_h_agg = code_h_agg.reset_index()

va_pd = va_pd.merge(code_cat_h_agg, on=["code", "sub_category", "horizon"], how="left")
va_pd = va_pd.merge(code_h_agg, on=["code", "horizon"], how="left")
va_pd = va_pd.fillna(global_mean)

# Historical aggregate predictions
iter9_preds = np.zeros(len(va_pd))
for h, (w_cm, w_cmed, w_chm, w_chmed) in [
    (1, (0.5, 0.3, 0.0, 0.2)),
    (3, (0.4, 0.4, 0.0, 0.2)),
    (10, (0.3, 0.4, 0.0, 0.3)),
    (25, (0.2, 0.3, 0.0, 0.5)),
]:
    mask = (va_pd["horizon"] == h).to_numpy()
    pred = (
        w_cm * va_pd.loc[mask, "code_cat_h_mean"].values
        + w_cmed * va_pd.loc[mask, "code_cat_h_median"].values
        + w_chm * va_pd.loc[mask, "code_h_mean"].values
        + w_chmed * va_pd.loc[mask, "code_h_median"].values
    )
    iter9_preds[mask] = pred

iter9_score, _ = weighted_rmse_score(
    va_pd["y_target"].values, iter9_preds, va_pd["weight"].fillna(1.0).values
)
print(f"Historical aggregates score: {iter9_score:.4f}")

# Second LightGBM with different seed
print("\n" + "=" * 70)
print("TRAINING SECOND LIGHTGBM (DIFFERENT SEED)")
print("=" * 70)

lgb_preds_va_v2 = np.zeros(len(va_pd))
lgb_preds_test_v2 = np.zeros(len(test_pd))

for h in horizons:
    tr_h = tr_pd[tr_pd["horizon"] == h]
    va_h_mask = va_pd["horizon"] == h
    test_h_mask = test_pd["horizon"] == h

    if len(tr_h) == 0:
        continue

    sample_size = min(250000, len(tr_h))
    if len(tr_h) > sample_size:
        tr_h = tr_h.sample(n=sample_size, random_state=123)

    X_tr_h = tr_h[model_features].copy()
    y_tr_h = tr_h["y_target"].values
    w_tr_h = tr_h["weight"].fillna(1.0).values

    for cf in cat_features:
        X_tr_h[cf] = X_tr_h[cf].astype("category")

    X_va_h = va_pd.loc[va_h_mask, model_features].copy()
    y_va_h = va_pd.loc[va_h_mask, "y_target"].values
    w_va_h = va_pd.loc[va_h_mask, "weight"].fillna(1.0).values

    for cf in cat_features:
        X_va_h[cf] = X_va_h[cf].astype("category")

    X_test_h = test_pd.loc[test_h_mask, model_features].copy()
    for cf in cat_features:
        X_test_h[cf] = X_test_h[cf].astype("category")

    lgb_train = lgb.Dataset(X_tr_h, y_tr_h, weight=w_tr_h)
    lgb_valid = lgb.Dataset(X_va_h, y_va_h, weight=w_va_h, reference=lgb_train)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 127,
        "learning_rate": 0.03,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 3,
        "min_child_samples": 30,
        "verbose": -1,
        "seed": 123,
        "n_jobs": -1,
    }

    model = lgb.train(
        params, lgb_train, num_boost_round=2000,
        valid_sets=[lgb_valid],
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)],
    )

    if len(X_va_h) > 0:
        preds_va_h = model.predict(X_va_h)
        lgb_preds_va_v2[va_h_mask.to_numpy()] = preds_va_h

    if len(X_test_h) > 0:
        preds_test_h = model.predict(X_test_h)
        lgb_preds_test_v2[test_h_mask.to_numpy()] = preds_test_h
