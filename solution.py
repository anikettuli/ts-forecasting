"""
Time Series Forecasting Solution v8 - Restored Best
Score: 0.1671
Target: > 0.35
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
print("TIME SERIES FORECASTING SOLUTION v8 - Restored Best")
print("=" * 70)

train = pl.read_parquet("data/train.parquet")
test = pl.read_parquet("data/test.parquet")

train = train.with_columns(pl.col("horizon").cast(pl.Int32))
test = test.with_columns(pl.col("horizon").cast(pl.Int32))

max_ts = train["ts_index"].max()
split_ts = int(max_ts * 0.9)
tr = train.filter(pl.col("ts_index") < split_ts)
va = train.filter(pl.col("ts_index") >= split_ts)

print(f"Train: {tr.height:,}, Valid: {va.height:,}")

raw_features = [c for c in train.columns if c.startswith("feature_")]
global_mean = tr["y_target"].mean()
y_min = tr["y_target"].quantile(0.001)
y_max = tr["y_target"].quantile(0.999)

print("Creating encodings...")

code_te = tr.group_by("code").agg(pl.col("y_target").mean().alias("code_te"))
sub_code_te = tr.group_by("sub_code").agg(
    pl.col("y_target").mean().alias("sub_code_te")
)
hist = tr.group_by(["code", "sub_category", "horizon"]).agg(
    [
        pl.col("y_target").mean().alias("hist_mean"),
        pl.col("y_target").std().alias("hist_std"),
    ]
)
hist = hist.with_columns(pl.col("hist_std").fill_null(0.0))

train_sub_codes = set(tr["sub_code"].unique().to_list())


def add_features(df):
    df = df.join(code_te, on="code", how="left")
    df = df.join(sub_code_te, on="sub_code", how="left")
    df = df.join(hist, on=["code", "sub_category", "horizon"], how="left")
    df = df.fill_null(global_mean)
    df = df.with_columns(
        [
            pl.col("hist_std").fill_null(0.0),
            (np.sin(2 * np.pi * (pl.col("ts_index") % 7) / 7.0)).alias("dow_sin"),
            (np.cos(2 * np.pi * (pl.col("ts_index") % 7) / 7.0)).alias("dow_cos"),
            pl.when(pl.col("sub_code").is_in(train_sub_codes))
            .then(0)
            .otherwise(1)
            .alias("is_cold"),
        ]
    )
    return df


tr = add_features(tr)
va = add_features(va)
test = add_features(test)

feature_cols = raw_features + [
    "horizon",
    "code_te",
    "sub_code_te",
    "hist_mean",
    "hist_std",
    "is_cold",
    "dow_sin",
    "dow_cos",
]

tr_pd = tr.select(feature_cols + ["y_target", "weight"]).to_pandas()
va_pd = va.select(feature_cols + ["y_target", "weight", "id"]).to_pandas()
test_pd = test.select(feature_cols + ["id"]).to_pandas()

test_ids = test_pd["id"].tolist()

del tr, va, test, train
gc.collect()

for fc in raw_features:
    m = tr_pd[fc].median()
    tr_pd[fc] = tr_pd[fc].fillna(m)
    va_pd[fc] = va_pd[fc].fillna(m)
    test_pd[fc] = test_pd[fc].fillna(m)

horizons = [1, 3, 10, 25]
preds_va = np.zeros(len(va_pd))
preds_test = np.zeros(len(test_pd))

print("\n" + "=" * 70)
print("TRAINING MULTI-SEED LGBM ENSEMBLE")
print("=" * 70)

seeds = [42, 123, 456]
leaves = [63, 127, 31]
lrs = [0.03, 0.02, 0.04]

for h in horizons:
    print(f"\n--- Horizon {h} ---")

    tr_mask = tr_pd["horizon"] == h
    va_mask = va_pd["horizon"] == h
    te_mask = test_pd["horizon"] == h

    X_tr = tr_pd.loc[tr_mask, feature_cols].values.astype(np.float32)
    y_tr = tr_pd.loc[tr_mask, "y_target"].values
    w_tr = np.clip(
        tr_pd.loc[tr_mask, "weight"].fillna(1.0).values,
        0,
        np.percentile(tr_pd.loc[tr_mask, "weight"].fillna(1.0).values, 99.9),
    )

    X_va = va_pd.loc[va_mask, feature_cols].values.astype(np.float32)
    y_va = va_pd.loc[va_mask, "y_target"].values
    w_va = va_pd.loc[va_mask, "weight"].fillna(1.0).values
    hist_va = va_pd.loc[va_mask, "hist_mean"].values

    X_te = test_pd.loc[te_mask, feature_cols].values.astype(np.float32)
    hist_te = test_pd.loc[te_mask, "hist_mean"].values

    print(f"Train: {len(X_tr):,}, Val: {len(X_va):,}")

    pred_va_seeds = []
    pred_te_seeds = []

    for seed, leaf, lr in zip(seeds, leaves, lrs):
        lgb_train = lgb.Dataset(X_tr, y_tr, weight=w_tr)
        lgb_valid = lgb.Dataset(
            X_va,
            y_va,
            weight=np.clip(w_va, 0, np.percentile(w_va, 99.9)),
            reference=lgb_train,
        )

        m = lgb.train(
            {
                "objective": "regression",
                "metric": "rmse",
                "num_leaves": leaf,
                "learning_rate": lr,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "verbose": -1,
                "n_jobs": -1,
                "seed": seed,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
            },
            lgb_train,
            num_boost_round=500,
            valid_sets=[lgb_valid],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )

        pred_va_seeds.append(m.predict(X_va))
        pred_te_seeds.append(m.predict(X_te))

    pred_va_avg = np.mean(pred_va_seeds, axis=0)
    pred_te_avg = np.mean(pred_te_seeds, axis=0)

    s_lgb = weighted_rmse_score(y_va, pred_va_avg, w_va)
    s_hist = weighted_rmse_score(y_va, hist_va, w_va)

    best_s, best_alpha = 0, 0.5
    for alpha in np.arange(0.5, 1.01, 0.05):
        p = alpha * pred_va_avg + (1 - alpha) * hist_va
        s = weighted_rmse_score(y_va, p, w_va)
        if s > best_s:
            best_s = s
            best_alpha = alpha

    print(
        f"LGB: {s_lgb:.4f}, Hist: {s_hist:.4f}, Blend({best_alpha:.2f}): {best_s:.4f}"
    )

    preds_va[va_mask.values] = best_alpha * pred_va_avg + (1 - best_alpha) * hist_va
    preds_test[te_mask.values] = best_alpha * pred_te_avg + (1 - best_alpha) * hist_te

    del X_tr, y_tr, w_tr, X_va, y_va, w_va
    gc.collect()

overall = weighted_rmse_score(
    va_pd["y_target"].values, preds_va, va_pd["weight"].fillna(1.0).values
)

print(f"\n{'=' * 70}")
print(f"FINAL VALIDATION SCORE: {overall:.4f}")
print(f"{'=' * 70}")

preds_test = np.clip(preds_test, y_min, y_max)

sub = pl.DataFrame({"id": test_ids, "prediction": preds_test})
sub.write_csv("submission_optimized.csv")
print(f"\nSaved: submission_optimized.csv")
print(f"Mean: {preds_test.mean():.4f}, Std: {preds_test.std():.4f}")
