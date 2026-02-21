import polars as pl
import lightgbm as lgb
import numpy as np_cpu
import gc
import warnings
import os

warnings.filterwarnings("ignore")
pl.Config.set_streaming_chunk_size(10000)


def clear_memory():
    gc.collect()


def weighted_rmse_score(y_target, y_pred, w) -> float:
    """Official Kaggle Weighted RMSE Skill Score."""
    y_t = np_cpu.asarray(y_target)
    y_p = np_cpu.asarray(y_pred)
    weights = np_cpu.asarray(w)
    weights = np_cpu.clip(weights, 0, np_cpu.percentile(weights, 99.9))
    denom = np_cpu.sum(weights * y_t**2) + 1e-8
    ratio = np_cpu.sum(weights * (y_t - y_p) ** 2) / denom
    clipped = np_cpu.clip(ratio, 0.0, 1.0)
    score = np_cpu.sqrt(1.0 - clipped)
    return float(score)


def load_data():
    print("Loading datasets...")
    train_path = "data/train.parquet"
    test_path = "data/test.parquet"

    def optimize(df):
        opts = []
        for col, dtype in df.schema.items():
            if col == "id":
                continue
            if dtype == pl.Float64:
                opts.append(pl.col(col).cast(pl.Float32))
            elif dtype == pl.Int64:
                opts.append(pl.col(col).cast(pl.Int32))
            elif dtype in (pl.Utf8, pl.String):
                opts.append(pl.col(col).cast(pl.Categorical))
        return df.with_columns(opts)

    with pl.StringCache():
        tr_full = optimize(pl.read_parquet(train_path))
        te_df = optimize(pl.read_parquet(test_path))

    max_ts = tr_full["ts_index"].max()
    split_ts = max_ts - int((max_ts - tr_full["ts_index"].min()) * 0.1)

    tr_full = tr_full.with_columns(
        pl.when(pl.col("ts_index") < split_ts)
        .then(pl.lit("train"))
        .otherwise(pl.lit("valid"))
        .alias("split")
    )
    te_df = te_df.with_columns(pl.lit("test").alias("split"))

    if "y_target" not in te_df.columns:
        te_df = te_df.with_columns(pl.lit(0.0).alias("y_target").cast(pl.Float32))

    full_df = pl.concat([tr_full, te_df], how="diagonal")
    print(f"Loaded {full_df.shape[0]} total rows.")
    return full_df


def feature_engineering(full_df):
    print("Applying Target Encoding and Historical Blends...")
    train_only = full_df.filter(pl.col("split") == "train")

    # Target Encoding
    sub_code_te = train_only.group_by("sub_code").agg(
        pl.col("y_target").mean().alias("sub_code_te").cast(pl.Float32)
    )
    code_te = train_only.group_by("code").agg(
        pl.col("y_target").mean().alias("code_te").cast(pl.Float32)
    )
    sub_cat_te = train_only.group_by("sub_category").agg(
        pl.col("y_target").mean().alias("sub_cat_te").cast(pl.Float32)
    )
    global_mean = float(train_only["y_target"].mean())

    full_df = full_df.join(sub_code_te, on="sub_code", how="left")
    full_df = full_df.join(code_te, on="code", how="left")
    full_df = full_df.join(sub_cat_te, on="sub_category", how="left")

    full_df = full_df.with_columns(
        [
            pl.col("sub_code_te")
            .fill_null(pl.col("code_te"))
            .fill_null(pl.col("sub_cat_te"))
            .fill_null(global_mean),
            pl.col("code_te").fill_null(pl.col("sub_cat_te")).fill_null(global_mean),
            pl.col("sub_cat_te").fill_null(global_mean),
        ]
    )

    # Historical aggregations
    hist_agg = train_only.group_by(["code", "sub_category", "horizon"]).agg(
        [
            pl.col("y_target").mean().alias("hist_mean").cast(pl.Float32),
            pl.col("y_target").std().alias("hist_std").cast(pl.Float32),
        ]
    )
    full_df = full_df.join(hist_agg, on=["code", "sub_category", "horizon"], how="left")

    subcat_agg = train_only.group_by(["sub_category", "horizon"]).agg(
        [
            pl.col("y_target").mean().alias("sc_hist_mean").cast(pl.Float32),
            pl.col("y_target").std().alias("sc_hist_std").cast(pl.Float32),
        ]
    )
    full_df = full_df.join(subcat_agg, on=["sub_category", "horizon"], how="left")

    hor_agg = train_only.group_by("horizon").agg(
        [
            pl.col("y_target").mean().alias("h_hist_mean").cast(pl.Float32),
            pl.col("y_target").std().alias("h_hist_std").cast(pl.Float32),
        ]
    )
    full_df = full_df.join(hor_agg, on="horizon", how="left")

    full_df = full_df.with_columns(
        [
            pl.col("hist_mean")
            .fill_null(pl.col("sc_hist_mean"))
            .fill_null(pl.col("h_hist_mean"))
            .fill_null(0.0),
            pl.col("hist_std")
            .fill_null(pl.col("sc_hist_std"))
            .fill_null(pl.col("h_hist_std"))
            .fill_null(0.0),
        ]
    ).drop(["sc_hist_mean", "sc_hist_std", "h_hist_mean", "h_hist_std"])

    train_sub_codes = set(train_only["sub_code"].unique().to_list())
    full_df = full_df.with_columns(
        pl.when(pl.col("sub_code").is_in(train_sub_codes))
        .then(pl.lit(0))
        .otherwise(pl.lit(1))
        .alias("is_cold")
        .cast(pl.Int8)
    )

    full_df = full_df.with_columns(
        [
            (pl.col("ts_index") * 2 * np_cpu.pi / 7.0)
            .sin()
            .cast(pl.Float32)
            .alias("dow_sin"),
            (pl.col("ts_index") * 2 * np_cpu.pi / 7.0)
            .cos()
            .cast(pl.Float32)
            .alias("dow_cos"),
        ]
    )

    print("Applying Cross-Sectional Ranking Features...")
    important_vars = ["feature_al", "feature_am", "feature_cg", "feature_by", "weight"]
    cs_keys = ["ts_index", "horizon"]

    exprs = []
    # Rank percentile over the cross section
    for feat in important_vars:
        if feat in full_df.columns:
            exprs.append(
                (pl.col(feat).rank() / pl.count())
                .over(cs_keys)
                .alias(f"{feat}_rank_pct")
                .cast(pl.Float32)
            )
            # Market Z-Score
            exprs.append(
                (
                    (pl.col(feat) - pl.col(feat).mean().over(cs_keys))
                    / (pl.col(feat).std().over(cs_keys) + 1e-8)
                )
                .alias(f"{feat}_zscore")
                .cast(pl.Float32)
            )

    full_df = full_df.with_columns(exprs)
    clear_memory()

    return full_df


def train_and_predict():
    full_df = load_data()
    full_df = feature_engineering(full_df)

    exclude_cols = [
        "id",
        "code",
        "sub_code",
        "sub_category",
        "y_target",
        "weight",
        "split",
    ]
    features = [c for c in full_df.columns if c not in exclude_cols]
    print(f"Using {len(features)} features for modeling.")

    valid_df = full_df.filter(pl.col("split") == "valid")
    test_df = full_df.filter(pl.col("split") == "test")

    valid_preds = np_cpu.zeros(len(valid_df))
    test_preds = np_cpu.zeros(len(test_df))

    horizons = sorted(full_df["horizon"].unique().to_list())

    configs = [
        {"seed": 42, "leaves": 63, "lr": 0.03, "l1": 0.0, "l2": 5.0},
        {"seed": 123, "leaves": 127, "lr": 0.02, "l1": 0.5, "l2": 10.0},
        {"seed": 456, "leaves": 31, "lr": 0.04, "l1": 0.0, "l2": 5.0},
    ]

    for h in horizons:
        print(f"\nEvaluating Horizon {h}...")
        tr = full_df.filter((pl.col("split") == "train") & (pl.col("horizon") == h))
        va = valid_df.filter(pl.col("horizon") == h)
        te = test_df.filter(pl.col("horizon") == h)

        if tr.shape[0] == 0:
            continue

        X_tr = tr.select(features).fill_null(0).to_numpy()
        X_tr = np_cpu.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
        y_tr = tr["y_target"].to_numpy()

        w_raw = tr["weight"].fill_null(1.0).to_numpy()
        p999 = np_cpu.percentile(w_raw, 99.9)
        w_tr = np_cpu.clip(w_raw, 1e-3, p999)
        # Normalize weights to prevent GPU gradient overflow!
        w_tr = w_tr / np_cpu.mean(w_tr)

        X_va = va.select(features).fill_null(0).to_numpy()
        X_te = te.select(features).fill_null(0).to_numpy()
        X_va = np_cpu.nan_to_num(X_va, nan=0.0, posinf=0.0, neginf=0.0)
        X_te = np_cpu.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)

        m_preds_va = []
        m_preds_te = []

        for cfg in configs:
            model = lgb.LGBMRegressor(
                n_estimators=500,
                learning_rate=cfg["lr"],
                num_leaves=cfg["leaves"],
                random_state=cfg["seed"],
                reg_alpha=cfg["l1"],
                reg_lambda=cfg["l2"],
                device="gpu",
                gpu_use_dp=True,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                verbose=-1,
            )

            if va.shape[0] > 0:
                model.fit(
                    X_tr,
                    y_tr,
                    sample_weight=w_tr,
                    eval_set=[(X_va, va["y_target"].to_numpy())],
                    eval_sample_weight=[va["weight"].fill_null(1.0).to_numpy()],
                    callbacks=[lgb.early_stopping(30, verbose=False)],
                )
                m_preds_va.append(model.predict(X_va))
            else:
                model.fit(X_tr, y_tr, sample_weight=w_tr)

            if te.shape[0] > 0:
                m_preds_te.append(model.predict(X_te))

        if va.shape[0] > 0:
            lgb_pred_va = np_cpu.mean(m_preds_va, axis=0)
            hist_mean_va = va["hist_mean"].to_numpy()

            y_va = va["y_target"].to_numpy()
            w_va = va["weight"].fill_null(1.0).to_numpy()

            best_alpha = 1.0
            best_score = float("inf")

            for alpha in np_cpu.arange(0.5, 1.01, 0.05):
                blend_pred = alpha * lgb_pred_va + (1 - alpha) * hist_mean_va
                sc = weighted_rmse_score(y_va, blend_pred, w_va)
                if sc < best_score:
                    best_score = sc
                    best_alpha = alpha

            print(
                f"Optimal alpha for H={h}: {best_alpha:.2f} (Val Skill Score: {best_score:.4f})"
            )

            final_va = best_alpha * lgb_pred_va + (1 - best_alpha) * hist_mean_va
            h_idx = np_cpu.where((valid_df["horizon"] == h).to_numpy())[0]
            valid_preds[h_idx] = final_va
        else:
            best_alpha = 1.0

        if te.shape[0] > 0:
            lgb_pred_te = np_cpu.mean(m_preds_te, axis=0)
            hist_mean_te = te["hist_mean"].to_numpy()
            final_te = best_alpha * lgb_pred_te + (1 - best_alpha) * hist_mean_te
            h_idx_te = np_cpu.where((test_df["horizon"] == h).to_numpy())[0]
            test_preds[h_idx_te] = final_te

        clear_memory()

    overall_score = weighted_rmse_score(
        valid_df["y_target"].to_numpy(),
        valid_preds,
        valid_df["weight"].fill_null(1.0).to_numpy(),
    )
    print("\n" + "=" * 50)
    print(f"OVERALL VALIDATION SCORE: {overall_score:.4f}")
    print("=" * 50)

    print("Saving submission...")
    sub_df = test_df.select("id").with_columns(pl.Series("prediction", test_preds))

    original_test = pl.read_parquet("data/test.parquet").select("id")
    final_sub = original_test.join(sub_df, on="id", how="left").fill_null(0.0)
    final_sub.write_csv("submission.csv")
    print(f"Saved submission.csv with {final_sub.shape[0]} predictions.")


if __name__ == "__main__":
    train_and_predict()
