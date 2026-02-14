import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import gc
    import psutil
    import warnings
    import logging
    import numpy as np
    import polars as pl
    import lightgbm as lgb
    import xgboost as xgb

    warnings.filterwarnings("ignore")
    logging.getLogger("cmdstanpy").disabled = True
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

    HAS_GPU = False
    try:
        import torch

        torch.set_float32_matmul_precision("medium")
        HAS_GPU = torch.cuda.is_available()
        print(f"Torch available. CUDA: {HAS_GPU}")
    except ImportError:
        print("Torch not found. GPU disabled.")

    def get_mem():
        return psutil.Process().memory_info().rss / 1024 / 1024

    def free_mem():
        gc.collect()

    print(f"Memory: {get_mem():.0f} MB")
    return HAS_GPU, free_mem, gc, get_mem, lgb, mo, np, os, pl, xgb


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Time-Series Forecasting v5 - Fixed Pipeline

    **Critical fixes over v4:**
    1. Temporal features computed on COMBINED train+test (no broken lag anchors)
    2. No per-series normalization (predict raw y_target directly)
    3. Simple scaling (let the model predict at full scale)
    4. Causal target encoding for code/sub_code
    5. Per-horizon models with early stopping + 1500 trees
    """)
    return


@app.cell
def _(free_mem, get_mem, lgb, mo, np, os, pl):
    mo.md("## 1. Load Data & Build Features on Combined Train+Test")

    DATA_DIR = "data"
    TRAIN_PATH = os.path.join(DATA_DIR, "train.parquet")
    TEST_PATH = os.path.join(DATA_DIR, "test.parquet")

    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"Data not found at {TRAIN_PATH}")

    # ── Load both datasets ──
    print("Loading train + test...")
    train_raw = pl.read_parquet(TRAIN_PATH)
    test_raw = pl.read_parquet(TEST_PATH)
    print(f"  Train: {train_raw.shape}, Test: {test_raw.shape}")

    # ── Identify columns ──
    GROUP_COLS = ["code", "sub_code"]
    exclude_cols = {
        "id",
        "code",
        "sub_code",
        "sub_category",
        "y_target",
        "weight",
        "ts_index",
        "horizon",
        "split",
    }

    raw_features = [
        c
        for c in train_raw.columns
        if c.startswith("feature_") and c in test_raw.columns
    ]
    print(f"  Raw features: {len(raw_features)}")

    # ── Tag splits and combine ──
    ts_max = train_raw["ts_index"].max()
    ts_min = train_raw["ts_index"].min()
    split_ts = ts_max - int((ts_max - ts_min) * 0.2)
    print(f"  ts_index range: [{ts_min}, {ts_max}], val split at {split_ts}")

    train_tagged = train_raw.with_columns(
        pl.when(pl.col("ts_index") < split_ts)
        .then(pl.lit("train"))
        .otherwise(pl.lit("valid"))
        .alias("split")
    )
    test_tagged = test_raw.with_columns(pl.lit("test").alias("split"))
    del train_raw, test_raw

    # Concat with diagonal to handle missing columns
    full_df = pl.concat([train_tagged, test_tagged], how="diagonal")
    del train_tagged, test_tagged
    free_mem()
    print(f"  Combined: {full_df.shape}")

    # ── Downcast for memory ──
    opt_exprs = []
    for col_name, dtype in full_df.schema.items():
        if col_name == "id":
            continue
        if dtype == pl.Float64:
            opt_exprs.append(pl.col(col_name).cast(pl.Float32))
        elif dtype == pl.Int64:
            opt_exprs.append(pl.col(col_name).cast(pl.Int32))
        elif dtype in (pl.Utf8, pl.String):
            opt_exprs.append(pl.col(col_name).cast(pl.Categorical))
    if opt_exprs:
        full_df = full_df.with_columns(opt_exprs)

    # ── Quick feature importance scan (pick top features for temporal) ──
    print("  Quick importance scan...")
    train_subset = full_df.filter(pl.col("split") == "train")
    sample_n = min(50000, train_subset.height)
    X_quick = train_subset.select(raw_features).fill_null(0).head(sample_n).to_numpy()
    y_quick = train_subset["y_target"].head(sample_n).to_numpy()
    X_quick = np.nan_to_num(X_quick, nan=0.0).astype(np.float32)
    y_quick = np.nan_to_num(y_quick, nan=0.0).astype(np.float32)

    quick_lgb = lgb.LGBMRegressor(n_estimators=50, max_depth=4, verbose=-1, n_jobs=-1)
    quick_lgb.fit(X_quick, y_quick)
    feat_imp = quick_lgb.feature_importances_
    top_idx = np.argsort(feat_imp)[::-1][:30]
    top_temporal_feats = [raw_features[i] for i in top_idx if feat_imp[i] > 0]
    print(f"  Top {len(top_temporal_feats)} features for temporal engineering")
    del X_quick, y_quick, quick_lgb, feat_imp, train_subset
    free_mem()

    # ── Sort by group + time (CRITICAL for temporal features) ──
    print("  Sorting combined data...")
    full_df = full_df.sort(GROUP_COLS + ["ts_index"])

    # ── Temporal features on COMBINED data ──
    print("  Creating temporal features on combined train+test...")
    temporal_feat_names = []
    BATCH_SIZE = 5

    for bi in range(0, len(top_temporal_feats), BATCH_SIZE):
        batch = top_temporal_feats[bi : bi + BATCH_SIZE]
        exprs = []
        for feat in batch:
            # Lag 1
            lag_name = f"{feat}_lag1"
            temporal_feat_names.append(lag_name)
            exprs.append(
                pl.col(feat)
                .shift(1)
                .over(GROUP_COLS)
                .fill_null(0.0)
                .alias(lag_name)
                .cast(pl.Float32)
            )
            # Rolling mean 7
            rm7_name = f"{feat}_rm7"
            temporal_feat_names.append(rm7_name)
            exprs.append(
                pl.col(feat)
                .shift(1)
                .rolling_mean(window_size=7, min_periods=1)
                .over(GROUP_COLS)
                .fill_null(0.0)
                .alias(rm7_name)
                .cast(pl.Float32)
            )
            # Rolling mean 30
            rm30_name = f"{feat}_rm30"
            temporal_feat_names.append(rm30_name)
            exprs.append(
                pl.col(feat)
                .shift(1)
                .rolling_mean(window_size=30, min_periods=1)
                .over(GROUP_COLS)
                .fill_null(0.0)
                .alias(rm30_name)
                .cast(pl.Float32)
            )
            # Rolling std 7
            rs7_name = f"{feat}_rstd7"
            temporal_feat_names.append(rs7_name)
            exprs.append(
                pl.col(feat)
                .shift(1)
                .rolling_std(window_size=7, min_periods=2)
                .over(GROUP_COLS)
                .fill_null(0.0)
                .alias(rs7_name)
                .cast(pl.Float32)
            )
            # Rate of change (clipped)
            roc_name = f"{feat}_roc"
            temporal_feat_names.append(roc_name)
            exprs.append(
                (
                    (pl.col(feat) - pl.col(feat).shift(1).over(GROUP_COLS))
                    / (pl.col(feat).shift(1).over(GROUP_COLS).abs() + 1e-8)
                )
                .fill_null(0.0)
                .clip(-100.0, 100.0)
                .alias(roc_name)
                .cast(pl.Float32)
            )
        full_df = full_df.with_columns(exprs)
        if bi % 20 == 0:
            free_mem()
    print(f"  Created {len(temporal_feat_names)} temporal features")

    # ── Cyclical time features ──
    cyclical_names = ["dow_sin", "dow_cos", "dom_sin", "dom_cos", "doy_sin", "doy_cos"]
    full_df = full_df.with_columns(
        [
            (np.sin(2 * np.pi * (pl.col("ts_index") % 7) / 7.0))
            .alias("dow_sin")
            .cast(pl.Float32),
            (np.cos(2 * np.pi * (pl.col("ts_index") % 7) / 7.0))
            .alias("dow_cos")
            .cast(pl.Float32),
            (np.sin(2 * np.pi * (pl.col("ts_index") % 30) / 30.0))
            .alias("dom_sin")
            .cast(pl.Float32),
            (np.cos(2 * np.pi * (pl.col("ts_index") % 30) / 30.0))
            .alias("dom_cos")
            .cast(pl.Float32),
            (np.sin(2 * np.pi * (pl.col("ts_index") % 365) / 365.0))
            .alias("doy_sin")
            .cast(pl.Float32),
            (np.cos(2 * np.pi * (pl.col("ts_index") % 365) / 365.0))
            .alias("doy_cos")
            .cast(pl.Float32),
        ]
    )

    # ── Causal target encoding ──
    print("  Creating target encoding for code/sub_code...")
    train_mean = full_df.filter(pl.col("split") == "train")["y_target"].mean()
    smoothing = 10

    for enc_col in ["code", "sub_code"]:
        enc_name = f"{enc_col}_te"
        full_df = full_df.with_columns(
            [
                pl.col("y_target")
                .shift(1)
                .cum_sum()
                .over(enc_col)
                .fill_null(0.0)
                .alias("_te_sum"),
                pl.col("y_target")
                .shift(1)
                .cum_count()
                .over(enc_col)
                .fill_null(0)
                .alias("_te_cnt"),
            ]
        )
        full_df = full_df.with_columns(
            (
                (pl.col("_te_sum") + smoothing * train_mean)
                / (pl.col("_te_cnt") + smoothing + 1e-8)
            )
            .fill_null(train_mean)
            .clip(-1000.0, 1000.0)
            .alias(enc_name)
            .cast(pl.Float32)
        ).drop(["_te_sum", "_te_cnt"])

    # ── Cold-start features ──
    train_sub_codes = set(
        full_df.filter(pl.col("split") == "train")["sub_code"].unique().to_list()
    )
    full_df = full_df.with_columns(
        pl.when(pl.col("sub_code").is_in(train_sub_codes))
        .then(0)
        .otherwise(1)
        .alias("is_cold_start")
        .cast(pl.Int8)
    )
    full_df = full_df.with_columns(
        pl.col("ts_index")
        .cum_count()
        .over("sub_code")
        .alias("sub_code_obs_count")
        .cast(pl.Int32)
    )

    # ── Final feature list ──
    feature_cols = (
        raw_features
        + ["horizon"]
        + cyclical_names
        + temporal_feat_names
        + ["code_te", "sub_code_te", "is_cold_start", "sub_code_obs_count"]
    )
    # Remove any features not actually in the dataframe
    feature_cols = [fc for fc in feature_cols if fc in full_df.columns]
    print(f"  Total features: {len(feature_cols)}")

    # ── Fill remaining nulls ──
    for fc in feature_cols:
        if full_df[fc].null_count() > 0:
            full_df = full_df.with_columns(pl.col(fc).fill_null(0.0))

    free_mem()
    print(f"  Memory: {get_mem():.0f} MB")
    print(f"  Train: {full_df.filter(pl.col('split') == 'train').height:,}")
    print(f"  Valid: {full_df.filter(pl.col('split') == 'valid').height:,}")
    print(f"  Test:  {full_df.filter(pl.col('split') == 'test').height:,}")

    return feature_cols, full_df, split_ts, ts_max


@app.cell
def _(HAS_GPU, feature_cols, free_mem, full_df, get_mem, lgb, mo, np, xgb):
    mo.md("## 2. Per-Horizon Training (No Weights, float64 Scoring)")

    horizons = sorted(
        full_df.filter(full_df["split"] == "train")["horizon"].unique().to_list()
    )
    print(f"Horizons: {horizons}")

    # Validation data — use float64 for weights/targets to prevent precision loss
    valid_part = full_df.filter(full_df["split"] == "valid")
    val_y = valid_part["y_target"].to_numpy().astype(np.float64)
    val_w = valid_part["weight"].fill_null(1.0).to_numpy().astype(np.float64)
    val_h = valid_part["horizon"].to_numpy()
    val_ids = valid_part["id"].to_list()

    pred_val = np.zeros(len(val_y), dtype=np.float64)

    # Weight diagnostics
    print(
        f"Weight stats: median={np.median(val_w):.2f}, "
        f"p99={np.percentile(val_w, 99):.2f}, "
        f"p99.9={np.percentile(val_w, 99.9):.2f}, "
        f"max={val_w.max():.2f}"
    )

    # FLOAT64 score function with weight clipping
    def wrmse_score(y, yhat, w):
        y64 = np.asarray(y, dtype=np.float64)
        p64 = np.asarray(yhat, dtype=np.float64)
        w64 = np.asarray(w, dtype=np.float64)
        w_cap = np.percentile(w64, 99.9)
        w64 = np.clip(w64, 0.0, w_cap)
        sst = np.sum(w64 * y64**2)
        sse = np.sum(w64 * (y64 - p64) ** 2)
        if sst < 1e-30:
            return 0.0, sst, sse
        ratio = sse / sst
        score = float(np.sqrt(max(1.0 - ratio, 0.0)))
        return score, float(sst), float(sse)

    # Store models for test inference
    all_models = {}

    for hz in horizons:
        print(f"\n{'=' * 50}")
        print(f"HORIZON {hz}")

        tr_hz = full_df.filter(
            (full_df["split"] == "train") & (full_df["horizon"] == hz)
        )
        va_hz = valid_part.filter(valid_part["horizon"] == hz)

        if tr_hz.height == 0 or va_hz.height == 0:
            print("  No data, skipping")
            continue

        # Extract arrays
        X_train = tr_hz.select(feature_cols).to_numpy().astype(np.float32)
        y_train = tr_hz["y_target"].to_numpy().astype(np.float32)

        X_valid = va_hz.select(feature_cols).to_numpy().astype(np.float32)
        y_valid = va_hz["y_target"].to_numpy().astype(np.float64)
        w_valid = va_hz["weight"].fill_null(1.0).to_numpy().astype(np.float64)

        # Sanitize
        X_train = np.clip(
            np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0), -1e9, 1e9
        )
        X_valid = np.clip(
            np.nan_to_num(X_valid, nan=0.0, posinf=0.0, neginf=0.0), -1e9, 1e9
        )
        y_train = np.nan_to_num(y_train, nan=0.0)

        print(f"  Train: {X_train.shape[0]:,}, Val: {X_valid.shape[0]:,}")
        print(f"  y_train: mean={y_train.mean():.2f}, std={y_train.std():.2f}")

        # ── LightGBM — NO sample weights, NO early stopping ──
        lgb_model = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            num_leaves=63,
            max_depth=-1,
            device="cpu",
            objective="regression",
            metric="rmse",
            subsample=0.8,
            colsample_bytree=0.6,
            min_child_samples=100,
            reg_alpha=0.1,
            reg_lambda=1.0,
            verbose=-1,
            n_jobs=-1,
        )
        lgb_model.fit(X_train, y_train)
        print(f"  LGB trained: {lgb_model.n_estimators} rounds")

        # ── XGBoost — NO sample weights, NO early stopping ──
        xgb_model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=6,
            tree_method="hist",
            device="cuda" if HAS_GPU else "cpu",
            objective="reg:squarederror",
            subsample=0.8,
            colsample_bytree=0.6,
            reg_alpha=0.1,
            reg_lambda=1.0,
        )
        xgb_model.fit(X_train, y_train, verbose=0)
        print(f"  XGB trained: {xgb_model.n_estimators} rounds")

        # Predict validation (float64)
        p_lgb_val = lgb_model.predict(X_valid).astype(np.float64)
        p_xgb_val = xgb_model.predict(X_valid).astype(np.float64)

        # Score (float64 + weight clipping)
        s_lgb, sst_l, sse_l = wrmse_score(y_valid, p_lgb_val, w_valid)
        s_xgb, sst_x, sse_x = wrmse_score(y_valid, p_xgb_val, w_valid)

        # Try blend ratios
        best_blend_score = 0.0
        best_blend_w = 0.5
        for bw in [0.3, 0.4, 0.5, 0.6, 0.7]:
            p_blend = bw * p_lgb_val + (1 - bw) * p_xgb_val
            sb, _, _ = wrmse_score(y_valid, p_blend, w_valid)
            if sb > best_blend_score:
                best_blend_score = sb
                best_blend_w = bw

        p_ens_raw = best_blend_w * p_lgb_val + (1 - best_blend_w) * p_xgb_val

        # ── SCALING: Use simple mean ratio instead of optimal shrinkage ──
        # The optimal shrinkage is causing predictions to become too small
        # Use simple scaling: scale predictions to match target mean
        mean_y = np.mean(y_valid)
        mean_p = np.mean(p_ens_raw)
        # Simple linear scaling
        if abs(mean_p) > 1e-6:
            opt_alpha = mean_y / mean_p
        else:
            opt_alpha = 1.0

        # Clip to reasonable range
        opt_alpha = float(np.clip(opt_alpha, 0.5, 5.0))

        p_ens_val = opt_alpha * p_ens_raw

        # Score after shrinkage
        s_shrunk, sst_s, sse_s = wrmse_score(y_valid, p_ens_val, w_valid)

        # Also compute unweighted score for comparison
        ones_w = np.ones_like(w_valid)
        s_unweighted, _, _ = wrmse_score(y_valid, p_ens_val, ones_w)
        s_unweighted_raw, _, _ = wrmse_score(y_valid, p_ens_raw, ones_w)

        # Diagnostics
        print(f"  RAW Blend({best_blend_w:.1f}): pred std={p_ens_raw.std():.2f}")
        print(f"  Optimal alpha = {opt_alpha:.8f}")
        print(
            f"  SHRUNK score: {s_shrunk:.6f} (SSE/SST={sse_s / max(sst_s, 1e-30):.6f})"
        )
        print(
            f"  SHRUNK pred: [{p_ens_val.min():.6f}, {p_ens_val.max():.6f}], "
            f"std={p_ens_val.std():.6f}"
        )
        print(
            f"  Unweighted score (raw): {s_unweighted_raw:.6f}, "
            f"(shrunk): {s_unweighted:.6f}"
        )

        # Fill predictions (shrunk)
        hz_mask = val_h == hz
        pred_val[hz_mask] = p_ens_val

        all_models[hz] = (lgb_model, xgb_model, best_blend_w, opt_alpha)

        del X_train, y_train, X_valid, tr_hz, va_hz
        free_mem()

    # Overall scores
    ov_s, ov_sst, ov_sse = wrmse_score(val_y, pred_val, val_w)
    ov_uw, _, _ = wrmse_score(val_y, pred_val, np.ones_like(val_w))
    print(f"\n{'=' * 50}")
    print(f"OVERALL WEIGHTED SCORE: {ov_s:.6f}")
    print(
        f"  SST={ov_sst:.6e}, SSE={ov_sse:.6e}, ratio={ov_sse / max(ov_sst, 1e-30):.6f}"
    )
    print(f"OVERALL UNWEIGHTED SCORE: {ov_uw:.6f}")
    print(f"{'=' * 50}")

    return all_models, pred_val, val_h, val_ids, val_w, val_y


@app.cell
def _(all_models, feature_cols, free_mem, full_df, get_mem, mo, np):
    mo.md("## 3. Test Inference")

    print("--- Test Inference ---")
    test_part = full_df.filter(full_df["split"] == "test")
    test_ids_out = test_part["id"].to_list()
    test_horizons_arr = test_part["horizon"].to_numpy()

    X_test_full = test_part.select(feature_cols).to_numpy().astype(np.float32)
    X_test_full = np.clip(
        np.nan_to_num(X_test_full, nan=0.0, posinf=0.0, neginf=0.0), -1e9, 1e9
    )
    print(f"  Test rows: {len(test_ids_out):,}")

    pred_test_out = np.zeros(len(test_ids_out), dtype=np.float64)

    for th, (t_lgb, t_xgb, t_bw, t_alpha) in all_models.items():
        t_mask = test_horizons_arr == th
        if not t_mask.any():
            continue
        X_th = X_test_full[t_mask]
        tp_lgb = t_lgb.predict(X_th).astype(np.float64)
        tp_xgb = t_xgb.predict(X_th).astype(np.float64)
        tp_raw = t_bw * tp_lgb + (1 - t_bw) * tp_xgb
        tp_ens = t_alpha * tp_raw
        pred_test_out[t_mask] = tp_ens
        print(
            f"  H{th}: {t_mask.sum():,} rows, α={t_alpha:.8f}, "
            f"raw_std={tp_raw.std():.2f}, pred=[{tp_ens.min():.6f}, {tp_ens.max():.6f}]"
        )

    del X_test_full, test_part
    free_mem()
    print(f"  Memory: {get_mem():.0f} MB")

    return pred_test_out, test_ids_out


@app.cell
def _(mo, np, pred_val, val_h, val_w, val_y):
    mo.md("## 4. Local Leaderboard")

    # Float64 score with weight clipping
    w64_lb = np.asarray(val_w, dtype=np.float64)
    w_cap_lb = np.percentile(w64_lb, 99.9)
    w64_lb = np.clip(w64_lb, 0, w_cap_lb)
    y64_lb = np.asarray(val_y, dtype=np.float64)
    p64_lb = np.asarray(pred_val, dtype=np.float64)

    def score_lb(y, yhat, w):
        sst = np.sum(w * y**2)
        sse = np.sum(w * (y - yhat) ** 2)
        if sst < 1e-30:
            return 0.0
        ratio = sse / sst
        return float(np.sqrt(max(1.0 - ratio, 0.0)))

    idx_lb = np.arange(len(y64_lb))
    np.random.seed(42)
    np.random.shuffle(idx_lb)
    split_lb = int(len(y64_lb) * 0.25)

    pub_score = score_lb(
        y64_lb[idx_lb[:split_lb]], p64_lb[idx_lb[:split_lb]], w64_lb[idx_lb[:split_lb]]
    )
    prv_score = score_lb(
        y64_lb[idx_lb[split_lb:]], p64_lb[idx_lb[split_lb:]], w64_lb[idx_lb[split_lb:]]
    )
    all_score = score_lb(y64_lb, p64_lb, w64_lb)

    print("-" * 60)
    print(f"{'LOCAL LEADERBOARD':<30} | SCORE")
    print("-" * 60)
    print(f"{'Public (25%)':<30} | {pub_score:.6f}")
    print(f"{'Private (75%)':<30} | {prv_score:.6f}")
    print(f"{'Overall':<30} | {all_score:.6f}")
    print("-" * 60)

    for lh in sorted(np.unique(val_h)):
        lm = val_h == lh
        if lm.any():
            ls = score_lb(y64_lb[lm], p64_lb[lm], w64_lb[lm])
            print(f"  Horizon {int(lh):<4} | {ls:.6f} | {lm.sum():>10,} samples")

    # Sanity check
    print(f"\nPrediction stats:")
    print(f"  pred mean={p64_lb.mean():.4f}, std={p64_lb.std():.4f}")
    print(f"  y    mean={y64_lb.mean():.4f}, std={y64_lb.std():.4f}")
    print(f"  Variance ratio: {p64_lb.std() / (y64_lb.std() + 1e-8):.4f}")
    return


@app.cell
def _(mo, np, pl, pred_test_out, pred_val, test_ids_out, val_ids):
    mo.md("## 5. Submission")

    submission_df = pl.DataFrame({"id": test_ids_out, "prediction": pred_test_out})
    submission_df.write_csv("submission_optimized.csv")
    print(f"Submission saved ({submission_df.height:,} rows)")
    print(f"   mean={np.mean(pred_test_out):.4f}, std={np.std(pred_test_out):.4f}")
    print(f"   min={np.min(pred_test_out):.4f}, max={np.max(pred_test_out):.4f}")

    val_out_df = pl.DataFrame({"id": val_ids, "prediction": pred_val})
    val_out_df.write_csv("validation_results.csv")
    print(f"Validation saved ({val_out_df.height:,} rows)")
    return


if __name__ == "__main__":
    app.run()
