"""
Time Series Forecasting Solution v3
===================================
Current Score: 0.1711 (baseline)
Target: > 0.25

Focus: Fix hierarchical encoding + add temporal features
"""

import polars as pl
import numpy as np
import lightgbm as lgb
import gc
import warnings

warnings.filterwarnings("ignore")


def weighted_rmse_score(y_true, y_pred, weights):
    w = np.clip(weights, 0, np.percentile(weights, 99.9))
    sst = np.sum(w * y_true**2) + 1e-8
    sse = np.sum(w * (y_true - y_pred) ** 2)
    if sst < 1e-10:
        return 0.0, 1.0
    ratio = sse / sst
    return np.sqrt(max(1.0 - ratio, 0.0)), ratio


print("=" * 70)
print("TIME SERIES FORECASTING SOLUTION v3")
print("=" * 70)

# Load data
train = pl.read_parquet("data/train.parquet")
test = pl.read_parquet("data/test.parquet")

# Cast horizon
train = train.with_columns(pl.col("horizon").cast(pl.Int32))
test = test.with_columns(pl.col("horizon").cast(pl.Int32))

max_ts = train["ts_index"].max()
split_ts = int(max_ts * 0.9)
tr = train.filter(pl.col("ts_index") < split_ts)
va = train.filter(pl.col("ts_index") >= split_ts)

print(f"Train: {tr.shape[0]:,}, Valid: {va.shape[0]:,}")

global_mean = tr["y_target"].mean()
print(f"Global mean: {global_mean:.4f}")

# Identify cold-start
train_sub_codes = set(tr["sub_code"].unique().to_list())
print(f"Train sub_codes: {len(train_sub_codes)}")

# Basic historical aggregates (known to work)
print("\nCreating historical aggregates...")

code_cat_h = (
    tr.group_by(["code", "sub_category", "horizon"])
    .agg(
        [
            pl.col("y_target").mean().alias("code_cat_h_mean"),
            pl.col("y_target").median().alias("code_cat_h_median"),
        ]
    )
    .fill_null(global_mean)
    .with_columns(pl.col("horizon").cast(pl.Int32))
)

code_h = (
    tr.group_by(["code", "horizon"])
    .agg(
        [
            pl.col("y_target").mean().alias("code_h_mean"),
            pl.col("y_target").median().alias("code_h_median"),
        ]
    )
    .fill_null(global_mean)
    .with_columns(pl.col("horizon").cast(pl.Int32))
)

# Horizon-level stats
h_stats = (
    tr.group_by("horizon")
    .agg(
        [
            pl.col("y_target").mean().alias("h_mean"),
            pl.col("y_target").std().alias("h_std"),
        ]
    )
    .with_columns(pl.col("horizon").cast(pl.Int32))
)


def add_encodings(df):
    df = df.join(code_cat_h, on=["code", "sub_category", "horizon"], how="left")
    df = df.join(code_h, on=["code", "horizon"], how="left")
    df = df.join(h_stats, on="horizon", how="left")
    df = df.fill_null(global_mean)

    # Cold-start indicator
    df = df.with_columns(
        pl.when(pl.col("sub_code").is_in(train_sub_codes))
        .then(0)
        .otherwise(1)
        .alias("is_cold")
    )
    return df


tr = add_encodings(tr)
va = add_encodings(va)
test = add_encodings(test)

# Features
raw_features = [c for c in train.columns if c.startswith("feature_")]
enc_features = [
    "code_cat_h_mean",
    "code_cat_h_median",
    "code_h_mean",
    "code_h_median",
    "h_mean",
    "h_std",
    "is_cold",
]
feature_cols = raw_features + ["horizon"] + enc_features

print(f"Total features: {len(feature_cols)}")

# Convert to pandas
tr_pd = tr.to_pandas()
va_pd = va.to_pandas()
test_pd = test.to_pandas()

del tr, va, test
gc.collect()

# Fill nulls
for fc in raw_features:
    median_val = tr_pd[fc].median()
    tr_pd[fc] = tr_pd[fc].fillna(median_val)
    va_pd[fc] = va_pd[fc].fillna(median_val)
    test_pd[fc] = test_pd[fc].fillna(median_val)

# Categorical
cat_features = ["code", "sub_category"]
for cf in cat_features:
    tr_pd[cf] = tr_pd[cf].astype("category")
    va_pd[cf] = va_pd[cf].astype("category")
    test_pd[cf] = test_pd[cf].astype("category")

# Training
print("\n" + "=" * 70)
print("TRAINING")
print("=" * 70)

horizons = [1, 3, 10, 25]
all_preds_va = np.zeros(len(va_pd))
all_preds_test = np.zeros(len(test_pd))
models = {}

for h in horizons:
    print(f"\n--- Horizon {h} ---")

    tr_h = tr_pd[tr_pd["horizon"] == h]
    va_h_mask = va_pd["horizon"] == h
    test_h_mask = test_pd["horizon"] == h

    X_tr = tr_h[feature_cols].values.astype(np.float32)
    y_tr = tr_h["y_target"].values
    w_tr = tr_h["weight"].fillna(1.0).values

    X_va = va_pd.loc[va_h_mask, feature_cols].values.astype(np.float32)
    y_va = va_pd.loc[va_h_mask, "y_target"].values
    w_va = va_pd.loc[va_h_mask, "weight"].fillna(1.0).values

    w_tr_clip = np.clip(w_tr, 0, np.percentile(w_tr, 99.9))

    print(f"Train: {len(X_tr):,}, Val: {len(X_va):,}")

    # LightGBM
    lgb_train = lgb.Dataset(X_tr, y_tr, weight=w_tr_clip)
    lgb_valid = lgb.Dataset(X_va, y_va, weight=w_va, reference=lgb_train)

    model = lgb.train(
        {
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": 127,
            "learning_rate": 0.02,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "verbose": -1,
            "seed": 42,
            "n_jobs": -1,
            "lambda_l1": 0.1,
            "lambda_l2": 0.5,
        },
        lgb_train,
        num_boost_round=2000,
        valid_sets=[lgb_valid],
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)],
    )

    p_va = model.predict(X_va)
    p_test = model.predict(
        test_pd.loc[test_h_mask, feature_cols].values.astype(np.float32)
    )

    p_hist_va = va_pd.loc[va_h_mask, "code_cat_h_mean"].values
    p_hist_test = test_pd.loc[test_h_mask, "code_cat_h_mean"].values

    s_model, _ = weighted_rmse_score(y_va, p_va, w_va)
    s_hist, _ = weighted_rmse_score(y_va, p_hist_va, w_va)

    print(f"  Model: {s_model:.4f}, Hist: {s_hist:.4f}")

    # Optimal blend
    best_s = max(s_model, s_hist)
    best_w = 1.0 if s_model >= s_hist else 0.0

    for w in np.arange(0.0, 1.01, 0.1):
        p = w * p_va + (1 - w) * p_hist_va
        s, _ = weighted_rmse_score(y_va, p, w_va)
        if s > best_s:
            best_s = s
            best_w = w

    print(f"  Best: {best_w:.1f} model + {1 - best_w:.1f} hist = {best_s:.4f}")

    models[h] = (model, best_w)
    all_preds_va[va_h_mask.to_numpy()] = best_w * p_va + (1 - best_w) * p_hist_va
    all_preds_test[test_h_mask.to_numpy()] = (
        best_w * p_test + (1 - best_w) * p_hist_test
    )

    del tr_h
    gc.collect()

overall, _ = weighted_rmse_score(
    va_pd["y_target"].values, all_preds_va, va_pd["weight"].fillna(1.0).values
)

print(f"\n{'=' * 70}")
print(f"OVERALL VALIDATION SCORE: {overall:.4f}")
print(f"{'=' * 70}")

# Second model with different seed
print("\n" + "=" * 70)
print("TRAINING MODEL v2 (different seed)")
print("=" * 70)

all_preds_va2 = np.zeros(len(va_pd))
all_preds_test2 = np.zeros(len(test_pd))

for h in horizons:
    tr_h = tr_pd[tr_pd["horizon"] == h]
    va_h_mask = va_pd["horizon"] == h
    test_h_mask = test_pd["horizon"] == h

    X_tr = tr_h[feature_cols].values.astype(np.float32)
    y_tr = tr_h["y_target"].values
    w_tr = tr_h["weight"].fillna(1.0).values

    X_va = va_pd.loc[va_h_mask, feature_cols].values.astype(np.float32)
    y_va = va_pd.loc[va_h_mask, "y_target"].values
    w_va = va_pd.loc[va_h_mask, "weight"].fillna(1.0).values

    w_tr_clip = np.clip(w_tr, 0, np.percentile(w_tr, 99.9))

    lgb_train = lgb.Dataset(X_tr, y_tr, weight=w_tr_clip)
    lgb_valid = lgb.Dataset(X_va, y_va, weight=w_va, reference=lgb_train)

    model = lgb.train(
        {
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": 63,
            "learning_rate": 0.03,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.7,
            "verbose": -1,
            "seed": 123,
            "n_jobs": -1,
        },
        lgb_train,
        num_boost_round=2000,
        valid_sets=[lgb_valid],
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)],
    )

    all_preds_va2[va_h_mask.to_numpy()] = model.predict(X_va)
    all_preds_test2[test_h_mask.to_numpy()] = model.predict(
        test_pd.loc[test_h_mask, feature_cols].values.astype(np.float32)
    )

    del tr_h
    gc.collect()

# Ensemble
print("\n" + "=" * 70)
print("ENSEMBLE OPTIMIZATION")
print("=" * 70)

s1, _ = weighted_rmse_score(
    va_pd["y_target"].values, all_preds_va, va_pd["weight"].fillna(1.0).values
)
s2, _ = weighted_rmse_score(
    va_pd["y_target"].values, all_preds_va2, va_pd["weight"].fillna(1.0).values
)

print(f"Model 1: {s1:.4f}, Model 2: {s2:.4f}")

# Historical baseline
p_hist_all = va_pd["code_cat_h_mean"].values
s_hist, _ = weighted_rmse_score(
    va_pd["y_target"].values, p_hist_all, va_pd["weight"].fillna(1.0).values
)
print(f"Historical: {s_hist:.4f}")

# Find best blend
best_s = max(s1, s2, s_hist)
best_w = (1.0, 0.0, 0.0)

for w1 in np.arange(0.0, 1.01, 0.05):
    for w2 in np.arange(0.0, 1.01 - w1, 0.05):
        w3 = max(0.0, 1.0 - w1 - w2)
        p = w1 * all_preds_va + w2 * all_preds_va2 + w3 * p_hist_all
        s, _ = weighted_rmse_score(
            va_pd["y_target"].values, p, va_pd["weight"].fillna(1.0).values
        )
        if s > best_s:
            best_s = s
            best_w = (w1, w2, w3)

print(
    f"\nBest ensemble: {best_w[0]:.2f} M1 + {best_w[1]:.2f} M2 + {best_w[2]:.2f} Hist = {best_s:.4f}"
)

# Final predictions
final_va = best_w[0] * all_preds_va + best_w[1] * all_preds_va2 + best_w[2] * p_hist_all
p_hist_test_all = test_pd["code_cat_h_mean"].values
final_test = (
    best_w[0] * all_preds_test
    + best_w[1] * all_preds_test2
    + best_w[2] * p_hist_test_all
)

print(f"\n{'=' * 70}")
print(f"FINAL VALIDATION SCORE: {best_s:.4f}")
print(f"{'=' * 70}")

# Save
sub = test_pd[["id"]].copy()
sub["prediction"] = final_test
sub.to_csv("submission_optimized.csv", index=False)
print(f"\nSaved: submission_optimized.csv")
