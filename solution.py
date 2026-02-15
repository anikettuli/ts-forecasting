"""
Time Series Forecasting Solution v5
===================================
Current Score: 0.1506
Target: > 0.25

Key improvements:
- Multi-seed ensemble (5 diverse configs)
- Horizon-specific models
- Historical aggregate baseline blend
- Proper weight clipping
"""

import polars as pl
import numpy as np
import lightgbm as lgb
import gc


def weighted_rmse_score(y_true, y_pred, weights):
    w = np.clip(weights, 0, np.percentile(weights, 99.9))
    sst = np.sum(w * y_true**2) + 1e-8
    sse = np.sum(w * (y_true - y_pred) ** 2)
    ratio = sse / sst
    return np.sqrt(max(1.0 - ratio, 0.0))


print("=" * 70)
print("TIME SERIES FORECASTING SOLUTION v5 - Multi-Seed Ensemble")
print("=" * 70)

train = pl.read_parquet("data/train.parquet")
test = pl.read_parquet("data/test.parquet")

train = train.with_columns(pl.col("horizon").cast(pl.Int32))
test = test.with_columns(pl.col("horizon").cast(pl.Int32))

max_ts = train["ts_index"].max()
split_ts = int(max_ts * 0.9)
tr = train.filter(pl.col("ts_index") < split_ts)
va = train.filter(pl.col("ts_index") >= split_ts)

print(f"Train: {tr.shape[0]:,}, Valid: {va.shape[0]:,}")

global_mean = tr["y_target"].mean()
raw_features = [c for c in train.columns if c.startswith("feature_")]

print("Creating historical aggregates...")
hist_cc = tr.group_by(["code", "sub_category", "horizon"]).agg(
    [
        pl.col("y_target").mean().alias("hist_mean"),
    ]
)
hist_cc = hist_cc.with_columns(pl.col("horizon").cast(pl.Int32))


def add_hist(df):
    return df.join(
        hist_cc, on=["code", "sub_category", "horizon"], how="left"
    ).fill_null(global_mean)


tr = add_hist(tr)
va = add_hist(va)
test = add_hist(test)

tr_pd = tr.to_pandas()
va_pd = va.to_pandas()
test_pd = test.to_pandas()

del tr, va, test
gc.collect()

for fc in raw_features:
    m = tr_pd[fc].median()
    tr_pd[fc] = tr_pd[fc].fillna(m)
    va_pd[fc] = va_pd[fc].fillna(m)
    test_pd[fc] = test_pd[fc].fillna(m)

feature_cols = raw_features + ["horizon", "hist_mean"]
horizons = [1, 3, 10, 25]

preds_va = np.zeros(len(va_pd))
preds_test = np.zeros(len(test_pd))

configs = [
    {"seed": 42, "leaves": 127, "lr": 0.02},
    {"seed": 123, "leaves": 63, "lr": 0.03},
    {"seed": 456, "leaves": 255, "lr": 0.015},
    {"seed": 789, "leaves": 127, "lr": 0.025},
    {"seed": 999, "leaves": 63, "lr": 0.04},
]

print("\n" + "=" * 70)
print("TRAINING MULTI-SEED ENSEMBLE")
print("=" * 70)

for h in horizons:
    print(f"\n--- Horizon {h} ---")

    tr_h = tr_pd[tr_pd["horizon"] == h]
    va_h_mask = va_pd["horizon"] == h
    test_h_mask = test_pd["horizon"] == h

    X_tr = tr_h[feature_cols].values.astype(np.float32)
    y_tr = tr_h["y_target"].values
    w_tr = np.clip(
        tr_h["weight"].fillna(1.0).values,
        0,
        np.percentile(tr_h["weight"].fillna(1.0).values, 99.9),
    )

    X_va = va_pd.loc[va_h_mask, feature_cols].values.astype(np.float32)
    y_va_h = va_pd.loc[va_h_mask, "y_target"].values
    w_va_h = va_pd.loc[va_h_mask, "weight"].fillna(1.0).values
    hist_va_h = va_pd.loc[va_h_mask, "hist_mean"].values

    X_test = test_pd.loc[test_h_mask, feature_cols].values.astype(np.float32)
    hist_test_h = test_pd.loc[test_h_mask, "hist_mean"].values

    pred_va_list = []
    pred_test_list = []

    for cfg in configs:
        lgb_train = lgb.Dataset(X_tr, y_tr, weight=w_tr)
        lgb_valid = lgb.Dataset(
            X_va,
            y_va_h,
            weight=np.clip(w_va_h, 0, np.percentile(w_va_h, 99.9)),
            reference=lgb_train,
        )

        m = lgb.train(
            {
                "objective": "regression",
                "metric": "rmse",
                "num_leaves": cfg["leaves"],
                "learning_rate": cfg["lr"],
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "verbose": -1,
                "n_jobs": -1,
                "seed": cfg["seed"],
            },
            lgb_train,
            num_boost_round=1000,
            valid_sets=[lgb_valid],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )
        pred_va_list.append(m.predict(X_va))
        pred_test_list.append(m.predict(X_test))

    pred_va_avg = np.mean(pred_va_list, axis=0)
    pred_test_avg = np.mean(pred_test_list, axis=0)

    best_s = 0
    best_alpha = 0
    for alpha in np.arange(0, 1.01, 0.05):
        p = alpha * pred_va_avg + (1 - alpha) * hist_va_h
        s = weighted_rmse_score(y_va_h, p, w_va_h)
        if s > best_s:
            best_s = s
            best_alpha = alpha

    print(
        f"Best blend: {best_alpha:.2f} LGB + {1 - best_alpha:.2f} hist = {best_s:.4f}"
    )

    preds_va[va_h_mask.to_numpy()] = (
        best_alpha * pred_va_avg + (1 - best_alpha) * hist_va_h
    )
    preds_test[test_h_mask.to_numpy()] = (
        best_alpha * pred_test_avg + (1 - best_alpha) * hist_test_h
    )

    del tr_h
    gc.collect()

overall = weighted_rmse_score(
    va_pd["y_target"].values, preds_va, va_pd["weight"].fillna(1.0).values
)

print(f"\n{'=' * 70}")
print(f"FINAL VALIDATION SCORE: {overall:.4f}")
print(f"{'=' * 70}")

sub = test_pd[["id"]].copy()
sub["prediction"] = preds_test
sub.to_csv("submission_optimized.csv", index=False)
print(f"\nSaved: submission_optimized.csv")
print(f"Predictions: mean={preds_test.mean():.4f}, std={preds_test.std():.4f}")
