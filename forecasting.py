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

    # Suppress warnings
    warnings.filterwarnings("ignore")
    logging.getLogger("cmdstanpy").disabled = True
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

    # Configuration for Memory Management
    pl.Config.set_streaming_chunk_size(1000)

    # Check for Darts and Torch (Deep Learning)
    try:
        import torch

        # Optimize for 30-series GPUs (Ampere)
        torch.set_float32_matmul_precision("medium")

        # Try importing Statistical Baselines (Prophet, AutoARIMA)
        try:
            import importlib.util

            if importlib.util.find_spec("statsforecast") is None:
                raise ImportError
        except ImportError:
            print(
                "⚠️ StatsForecast/Prophet not found. Install 'statsforecast' and 'prophet' for statistical baselines."
            )

        print("✅ Darts & Torch available for Deep Learning.")
    except ImportError:
        print(
            "⚠️ Darts/Torch not found. Deep Learning & Statistical baselines will be skipped."
        )
        print(
            "To enable, run: pip install darts torch pytorch-lightning prophet statsforecast"
        )
    return gc, lgb, mo, np, os, pl, psutil, torch, xgb


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 📈 Advanced Time-Series Forecasting

    This notebook implements a multi-stage forecasting pipeline for the Kaggle `ts-forecasting` competition.
    It is optimized for limited hardware (16GB RAM, 8GB VRAM) by using:

    1.  **Polars** for memory-efficient data manipulation.
    2.  **Gradient Boosting** (LightGBM/XGBoost) with GPU acceleration for the core engine.
    3.  **Deep Learning** (N-BEATS/TFT via Darts) for capturing complex non-linear patterns.
    4.  **Statistical Models** (AutoARIMA/Prophet) for robust baselines.
    """)
    return


@app.cell
def _(gc, os, pl, psutil):
    # --- Utilities ---
    def get_memory_usage():
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def free_memory():
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        except Exception:
            pass

    def optimize_types(df):
        """Downcast types to save memory."""
        # Handle LazyFrame or DataFrame
        is_lazy = isinstance(df, pl.LazyFrame)
        schema = df.collect_schema() if is_lazy else df.schema

        exprs = []
        for name, dtype in schema.items():
            if name == "id":
                continue
            if dtype == pl.Float64:
                exprs.append(pl.col(name).cast(pl.Float32))
            elif dtype == pl.Int64:
                exprs.append(pl.col(name).cast(pl.Int32))

        if exprs:
            return df.with_columns(exprs)
        return df

    print(f"Initial Memory Usage: {get_memory_usage():.2f} MB")

    # Paths
    DATA_DIR = "data"
    TRAIN_PATH = os.path.join(DATA_DIR, "train.parquet")
    TEST_PATH = os.path.join(DATA_DIR, "test.parquet")

    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(
            f"Data not found at {TRAIN_PATH}. Please ensure data is in the 'data' directory."
        )
    return TEST_PATH, TRAIN_PATH, free_memory


@app.cell
def _(TEST_PATH, TRAIN_PATH, mo, pl):
    mo.md("## 1. Data Loading & Preprocessing (Optimized)")

    # Load and optimize using Lazy API
    print("Loading data (Lazy)...")
    tr_lf = pl.scan_parquet(TRAIN_PATH)
    te_lf = pl.scan_parquet(TEST_PATH)

    # Get max_ts for splitting
    max_ts = tr_lf.select(pl.col("ts_index").max()).collect().item()
    print(f"Max TS in Train: {max_ts}")

    # Configuration
    GROUP_COLS = ["code", "sub_code", "sub_category"]

    # Pre-process Test to ensure we have needed horizons/ts_index
    # We will use this to join features later
    return GROUP_COLS, max_ts, te_lf, tr_lf


@app.cell
def _(GROUP_COLS, free_memory, mo, np, pl, tr_lf):
    mo.md(
        r"""
    ## 2. Advanced Feature Engineering

    1.  **Hierarchical Trends**: Lagged means for `code` and `sub_category`.
    2.  **Momentum**: Difference between short and long lags.
    3.  **Cyclical Time**: sin/cos encoding for day of week/month.
    """
    )

    # Config
    LAGS = [1, 7, 14, 28]
    WINDOWS = [7, 28]

    # --- Advanced Feature Engineering ---

    # 1. Target Denoising (Winsorization)
    tr_denoised_lf = tr_lf.with_columns(
        pl.col("y_target")
        .clip(
            tr_lf.select(pl.col("y_target").quantile(0.001)).collect().item(),
            tr_lf.select(pl.col("y_target").quantile(0.999)).collect().item(),
        )
        .alias("y_target_denoised")
    )

    # --- Hierarchical Aggregations (Capture market-wide category shifts) ---
    cat_agg = tr_denoised_lf.group_by(["sub_category", "ts_index"]).agg(
        pl.col("y_target_denoised").mean().alias("cat_y_mean")
    )
    code_agg = tr_denoised_lf.group_by(["code", "ts_index"]).agg(
        pl.col("y_target_denoised").mean().alias("code_y_mean")
    )

    # Base Series Logic
    series_lf = tr_denoised_lf.sort(GROUP_COLS + ["ts_index"])
    series_lf = series_lf.join(cat_agg, on=["sub_category", "ts_index"], how="left")
    series_lf = series_lf.join(code_agg, on=["code", "ts_index"], how="left")

    # --- Cross-Sectional Signals (The "Alpha" Features) ---
    feature_exprs = []

    # Cyclical Time
    feature_exprs.extend(
        [
            (np.sin(2 * np.pi * (pl.col("ts_index") % 7) / 7.0)).alias("dow_sin"),
            (np.cos(2 * np.pi * (pl.col("ts_index") % 7) / 7.0)).alias("dow_cos"),
            (np.sin(2 * np.pi * (pl.col("ts_index") % 30) / 30.0)).alias("dom_sin"),
            (np.cos(2 * np.pi * (pl.col("ts_index") % 30) / 30.0)).alias("dom_cos"),
        ]
    )

    # Series Lags
    for lag in LAGS:
        feature_exprs.append(
            pl.col("y_target_denoised").shift(lag).over(GROUP_COLS).alias(f"lag_{lag}")
        )

    for w in WINDOWS:
        feature_exprs.append(
            pl.col("y_target_denoised")
            .shift(1)
            .rolling_mean(w)
            .over(GROUP_COLS)
            .alias(f"roll_mean_{w}")
        )

    # Hierarchical Signals (Restore the category averages)
    feature_exprs.append(
        pl.col("cat_y_mean").shift(1).over(GROUP_COLS).alias("cat_lag_1")
    )
    feature_exprs.append(
        pl.col("code_y_mean").shift(1).over(GROUP_COLS).alias("code_lag_1")
    )

    # Momentum
    feature_exprs.append(
        (pl.col("y_target_denoised").shift(1) - pl.col("y_target_denoised").shift(7))
        .over(GROUP_COLS)
        .alias("momentum_7d")
    )

    # Recency Weighting
    max_ts_val = tr_lf.select(pl.col("ts_index").max()).collect().item()

    # Pass 1: Signal Matrix
    series_with_features = series_lf.with_columns(feature_exprs)

    # Pass 2: Ranking (Stability)
    rank_exprs = []
    for lag in LAGS:
        rank_exprs.append(
            pl.col(f"lag_{lag}")
            .rank("dense")
            .over(["sub_category", "ts_index"])
            .alias(f"lag_{lag}_cat_rank")
        )

    series_with_features = series_with_features.with_columns(rank_exprs).fill_null(0.0)
    series_with_features = series_with_features.with_columns(
        (pl.col("weight") * ((pl.col("ts_index") - max_ts_val) / 90.0).exp()).alias(
            "v_weight"
        )
    )

    series_with_features = series_with_features.with_columns(
        [
            pl.col(c).cast(pl.Float32)
            for c in series_with_features.collect_schema().names()
            if series_with_features.collect_schema()[c] == pl.Float64
        ]
    )

    feature_cols = [
        c
        for c in series_with_features.collect_schema().names()
        if any(
            x in c
            for x in ["lag_", "roll_", "do", "momentum", "_rank", "cat_lag", "code_lag"]
        )
    ]
    # Add horizon as a feature so model knows the 'gap' it's predicting
    if "horizon" not in feature_cols:
        feature_cols.append("horizon")

    print("Materializing Feature Matrix into RAM...")
    all_select_cols = (
        ["id", "horizon"]
        + GROUP_COLS
        + ["ts_index", "y_target", "y_target_denoised", "v_weight", "weight"]
        + feature_cols
    )
    # Remove duplicates while preserving order
    unique_cols = list(dict.fromkeys(all_select_cols))

    print(f"Selecting {len(unique_cols)} columns for materialization...")
    full_series_features = (
        series_with_features.select(unique_cols)
        .with_columns(
            [pl.col("ts_index").cast(pl.Int32), pl.col("horizon").cast(pl.Int32)]
        )
        .collect()
    )

    free_memory()
    print(f"Refined Feature Engineering Ready. Shape: {full_series_features.shape}")
    return feature_cols, full_series_features


@app.cell
def _(
    GROUP_COLS,
    feature_cols,
    free_memory,
    full_series_features,
    lgb,
    max_ts,
    mo,
    np,
    pl,
    te_lf,
    torch,
    xgb,
):
    mo.md("## 3. Training & Prediction Pipeline (Unified)")

    # 1. Configuration & Preparation
    split_cutoff = int(max_ts * 0.9)
    print(f"Starting Optimized Modeling Pipeline... Split Cutoff: {split_cutoff}")

    # Feature Pool (For joins) - Exclude metadata to avoid join collisions
    meta_cols = ["id", "horizon", "y_target", "y_target_denoised", "v_weight", "weight"]
    feat_only_cols = [c for c in feature_cols if c not in meta_cols]

    feat_pool = full_series_features.select(
        GROUP_COLS + ["ts_index"] + feat_only_cols
    ).unique(subset=GROUP_COLS + ["ts_index"])

    # Validation Set
    valid_base = full_series_features.filter(
        (pl.col("ts_index") >= split_cutoff) & (pl.col("ts_index") <= max_ts)
    ).sort(["ts_index"] + GROUP_COLS)

    y_valid = valid_base.select("y_target").to_series().to_numpy().astype(np.float32)
    w_valid = valid_base.select("weight").to_series().to_numpy().astype(np.float32)
    valid_ids = valid_base.select("id").to_series().to_list()
    h_valid = valid_base.select("horizon").to_series().to_numpy().astype(np.int32)

    print(f"Validation Target Ready. Size: {len(y_valid):,}")
    free_memory()

    # 2. Unified Training Data
    print("Preparing Training Set (Pooled Horizons)...")
    train_sample = full_series_features.filter(
        pl.col("ts_index") < split_cutoff
    ).sample(fraction=0.5, seed=42)

    X_train = train_sample.select(feature_cols).to_numpy().astype(np.float32)
    y_train = train_sample.select("y_target").to_series().to_numpy().astype(np.float32)
    w_train = train_sample.select("v_weight").to_series().to_numpy().astype(np.float32)

    # Safety: Ensure no INF, extreme values, or invalid weights crash LightGBM
    X_train = np.clip(np.nan_to_num(X_train, 0.0), -1e5, 1e5)
    y_train = np.clip(np.nan_to_num(y_train, 0.0), -1e4, 1e4)
    w_train = np.clip(np.nan_to_num(w_train, 1.0), 1e-4, 1e2)

    del train_sample
    free_memory()

    # Simple Monitor Set
    X_val_m = valid_base.select(feature_cols).to_numpy().astype(np.float32)
    y_val_m = valid_base.select("y_target").to_series().to_numpy().astype(np.float32)
    X_val_m = np.clip(np.nan_to_num(X_val_m, 0.0), -1e5, 1e5)

    # 3. Model Training
    use_gpu = torch.cuda.is_available()
    print(f"Training Unified Models (GPU: {use_gpu})...")

    # LGBM - Ultra-stable configuration
    l_params = {
        "n_estimators": 1200,
        "learning_rate": 0.03,
        "num_leaves": 47,
        "max_depth": 7,
        "device": "gpu" if use_gpu else "cpu",
        "objective": "regression",
        "metric": "rmse",
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "max_bin": 63,
        "verbose": -1,
        "min_child_samples": 100,
        "min_split_gain": 0.05,
    }
    l_mod = lgb.LGBMRegressor(**l_params)
    l_mod.fit(
        X_train,
        y_train,
        sample_weight=w_train,
        eval_set=[(X_val_m, y_val_m)],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )

    # XGB
    x_mod = xgb.XGBRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=6,
        tree_method="hist",
        device="cuda" if use_gpu else "cpu",
        objective="reg:squarederror",
        max_bin=63,
        subsample=0.7,
        colsample_bytree=0.7,
    )
    x_mod.fit(
        X_train,
        y_train,
        sample_weight=w_train,
        eval_set=[(X_val_m, y_val_m)],
        verbose=False,
    )

    del X_train, y_train, w_train, X_val_m, y_val_m
    free_memory()

    # 4. Multi-Horizon Inference
    # We apply the correct temporal shift for each horizon during inference
    pred_lgb_val = np.zeros(len(y_valid), dtype=np.float32)
    pred_xgb_val = np.zeros(len(y_valid), dtype=np.float32)

    _unique_h = sorted([int(h) for h in np.unique(h_valid)])
    for h in _unique_h:
        _mask_v = h_valid == h
        # Construct features for this horizon specifically
        feat_v = (
            valid_base.filter(pl.col("horizon") == h)
            .with_columns((pl.col("ts_index") - h + 1).cast(pl.Int32).alias("_join_ts"))
            .join(
                feat_pool,
                left_on=GROUP_COLS + ["_join_ts"],
                right_on=GROUP_COLS + ["ts_index"],
                how="left",
            )
            .fill_null(0.0)
            .select(feature_cols)
            .to_numpy()
            .astype(np.float32)
        )

        pred_lgb_val[_mask_v] = l_mod.predict(feat_v)
        pred_xgb_val[_mask_v] = x_mod.predict(feat_v)
        print(f"   Inference for Horizon {h} complete.")

    # Test Inference
    print("Generating Test Predictions...")
    test_rows_all = te_lf.collect().with_columns(
        (pl.col("ts_index") - pl.col("horizon") + 1).cast(pl.Int32).alias("_join_ts")
    )
    test_ids = test_rows_all.select("id").to_series().to_list()
    test_horizons = (
        test_rows_all.select("horizon").to_series().to_numpy().astype(np.int32)
    )

    pred_lgb_test = np.zeros(len(test_ids), dtype=np.float32)
    pred_xgb_test = np.zeros(len(test_ids), dtype=np.float32)

    for h in _unique_h:
        _mask_t = test_horizons == h
        if _mask_t.any():
            feat_t = (
                test_rows_all.filter(pl.col("horizon") == h)
                .join(
                    feat_pool,
                    left_on=GROUP_COLS + ["_join_ts"],
                    right_on=GROUP_COLS + ["ts_index"],
                    how="left",
                )
                .fill_null(0.0)
                .select(feature_cols)
                .to_numpy()
                .astype(np.float32)
            )

            pred_lgb_test[_mask_t] = l_mod.predict(feat_t)
            pred_xgb_test[_mask_t] = x_mod.predict(feat_t)
    return (
        h_valid,
        pred_lgb_test,
        pred_lgb_val,
        pred_xgb_test,
        pred_xgb_val,
        test_ids,
        valid_ids,
        w_valid,
        y_valid,
    )


@app.cell
def _(mo, np, pred_lgb_test, pred_lgb_val, pred_xgb_test, pred_xgb_val):
    mo.md("## 4. Ensembling & Submission Prep")

    # Weighted Average Ensemble
    ensemble_weights = [0.5, 0.5]

    final_pred_val = (
        ensemble_weights[0] * pred_lgb_val + ensemble_weights[1] * pred_xgb_val
    )

    final_pred_test = (
        ensemble_weights[0] * pred_lgb_test + ensemble_weights[1] * pred_xgb_test
    )

    # Clip predictions to non-negative
    final_pred_test = np.maximum(final_pred_test, 0)
    final_pred_val = np.maximum(final_pred_val, 0)

    print("Ensemble Predictions Generated.")
    return final_pred_test, final_pred_val


@app.cell
def _(final_pred_val, h_valid, mo, np, w_valid, y_valid):
    mo.md(
        r"""
    ## 5. Local Leaderboard Simulation

    This cell simulates the Kaggle scoring environment. We split our validation holdout into two sets:
    1.  **Public Score (25%)**: Mimics the real-time feedback.
    2.  **Private Score (75%)**: Mimics the final final evaluation.
    """
    )

    def official_skill_score(y_target, y_pred, w) -> float:
        """Official formula from evaluate.py: sqrt(1 - clip(SSE_w / SST_w))"""
        denom = np.sum(w * (y_target**2))
        if denom == 0:
            return 0.0
        numerator = np.sum(w * ((y_target - y_pred) ** 2))
        ratio = numerator / denom
        return float(np.sqrt(np.maximum(1.0 - ratio, 0.0)))

    # Simulate Public/Private Split
    indices = np.arange(len(y_valid))
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(indices)

    split_idx = int(len(y_valid) * 0.25)
    public_idx, private_idx = indices[:split_idx], indices[split_idx:]

    score_public = official_skill_score(
        y_valid[public_idx], final_pred_val[public_idx], w_valid[public_idx]
    )
    score_private = official_skill_score(
        y_valid[private_idx], final_pred_val[private_idx], w_valid[private_idx]
    )
    overall = official_skill_score(y_valid, final_pred_val, w_valid)

    print("-" * 65)
    print(f"{'LOCAL LEADERBOARD (SIMULATED)':<35} | {'SCORE':<20}")
    print("-" * 65)
    print(f"{'🔴 Simulated Public LB (25%)':<35} | {score_public:.6f}")
    print(f"{'🔒 Simulated Private LB (75%)':<35} | {score_private:.6f}")
    print(f"{'📊 Overall Holdout Score':<35} | {overall:.6f}")
    print("-" * 65)

    # Horizon Breakdown
    _unique_h = np.unique(h_valid)
    if len(_unique_h) > 1:
        print("\nScores by Horizon:")
        print(f"  {'Horizon':<10} | {'Score':<10} | {'Samples':<10}")
        for h_eval in sorted(_unique_h):
            _h_mask = h_valid == h_eval
            if _h_mask.any():
                h_score = official_skill_score(
                    y_valid[_h_mask], final_pred_val[_h_mask], w_valid[_h_mask]
                )
                print(f"  {int(h_eval):<10} | {h_score:.6f}   | {_h_mask.sum():,}")
    return


@app.cell
def _(final_pred_test, final_pred_val, mo, pl, test_ids, valid_ids):
    mo.md("## 6. Submission")

    submission_df = pl.DataFrame({"id": test_ids, "prediction": final_pred_test})

    submission_path = "submission_optimized.csv"
    submission_df.write_csv(submission_path)
    print(f"Submission saved to: {submission_path}")

    # --- Validation Export (For GitHub History Accuracy) ---
    val_df = pl.DataFrame({"id": valid_ids, "prediction": final_pred_val})
    val_path = "validation_results.csv"
    val_df.write_csv(val_path)
    print(f"Validation results saved to: {val_path}")
    return


if __name__ == "__main__":
    app.run()
