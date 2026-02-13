import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import sys
    import gc
    import time
    import psutil
    import warnings
    import logging
    import numpy as np
    import polars as pl
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.decomposition import IncrementalPCA

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
        from darts import TimeSeries
        from darts.models import NBEATSModel, TFTModel
        from darts.dataprocessing.transformers import Scaler

        # Try importing Statistical Baselines (Prophet, AutoARIMA)
        try:
            from darts.models import Prophet
            from darts.models import StatsForecastAutoARIMA

            stats_available = True
        except ImportError:
            stats_available = False
            print(
                "⚠️ StatsForecast/Prophet not found. Install 'statsforecast' and 'prophet' for statistical baselines."
            )

        models_available = True
        print("✅ Darts & Torch available for Deep Learning.")
    except ImportError:
        models_available = False
        stats_available = False
        print(
            "⚠️ Darts/Torch not found. Deep Learning & Statistical baselines will be skipped."
        )
        print(
            "To enable, run: pip install darts torch pytorch-lightning prophet statsforecast"
        )
    return (
        NBEATSModel,
        StatsForecastAutoARIMA,
        TimeSeries,
        gc,
        lgb,
        mo,
        models_available,
        np,
        os,
        pl,
        psutil,
        stats_available,
        torch,
        xgb,
    )


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
        except:
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
    return TEST_PATH, TRAIN_PATH, free_memory, get_memory_usage, optimize_types


@app.cell
def _(TEST_PATH, TRAIN_PATH, free_memory, mo, optimize_types, pl):
    mo.md("## 1. Data Loading & Preprocessing (Lazy Mode)")

    # Load and optimize using Lazy API
    print("Loading data (Lazy)...")
    _tr = pl.scan_parquet(TRAIN_PATH)
    _te = pl.scan_parquet(TEST_PATH)

    # Mark splits and optimize types lazily
    # Fetch just the max index to define split
    max_ts = _tr.select(pl.col("ts_index").max()).collect().item()
    _split_idx = int(max_ts * 0.9)

    _train_df = optimize_types(_tr.with_columns(pl.lit("train").alias("split")))
    _test_df = optimize_types(
        _te.with_columns(
            [
                pl.lit("test").alias("split"),
                pl.lit(0.0).alias("y_target"),
                pl.lit(1.0).alias("weight"),  # Dummy weight
            ]
        )
    )

    # Combine for feature engineering
    # Get common columns based on schema
    _tr_schema = _train_df.collect_schema()
    _te_schema = _test_df.collect_schema()
    _common_cols = [c for c in _tr_schema.names() if c in _te_schema.names()]

    full_df_raw = pl.concat(
        [_train_df.select(_common_cols), _test_df.select(_common_cols)]
    )

    # We do NOT collect here. Keep it lazy.
    del _tr, _te, _train_df, _test_df
    free_memory()

    print(f"Data Prepared (Lazy). Split Point: {_split_idx}")
    return full_df_raw, max_ts


@app.cell
def _(free_memory, full_df_raw, mo, pl):
    mo.md("## 2. Feature Engineering")

    # Config
    _LAGS = [1, 7, 14, 28]  # Daily, Weekly, Bi-weekly, Monthly
    _WINDOWS = [7, 28]

    print("Generating Features (Lazy)...")

    # Sort is required for window functions
    _df = full_df_raw.sort(["code", "sub_code", "ts_index"])

    _feature_exprs = []

    # 1. Lags
    for _lag in _LAGS:
        _feature_exprs.append(
            pl.col("y_target")
            .shift(_lag)
            .over(["code", "sub_code"])
            .alias(f"lag_{_lag}")
        )

    # 2. Rolling Statistics (Mean, Std)
    for _window in _WINDOWS:
        # We must shift by 1 to avoid leakage (using current target to predict current)
        _feature_exprs.append(
            pl.col("y_target")
            .shift(1)
            .rolling_mean(_window)
            .over(["code", "sub_code"])
            .alias(f"rolling_mean_{_window}")
        )
        _feature_exprs.append(
            pl.col("y_target")
            .shift(1)
            .rolling_std(_window)
            .over(["code", "sub_code"])
            .alias(f"rolling_std_{_window}")
        )

    # 3. Date Features (derived from ts_index assuming daily)
    _feature_exprs.append((pl.col("ts_index") % 7).alias("day_of_week"))
    _feature_exprs.append((pl.col("ts_index") % 30).alias("day_of_month"))

    # Apply Lazily
    full_df_featurized = _df.with_columns(_feature_exprs)

    # Fill Nulls created by lags (important for ML models)
    full_df_featurized = full_df_featurized.fill_nan(0.0).fill_null(0.0)

    # Clean up
    free_memory()
    print("Features Definition Created (Lazy).")

    # Define Feature Columns
    _exclude_cols = [
        "id",
        "y_target",
        "weight",
        "split",
        "code",
        "sub_code",
        "sub_category",
    ]
    # Inspect schema to get feature names
    _schema = full_df_featurized.collect_schema()
    feature_cols = [c for c in _schema.names() if c not in _exclude_cols]
    print(f"Feature Count: {len(feature_cols)}")
    return feature_cols, full_df_featurized


@app.cell
def _(
    feature_cols,
    free_memory,
    full_df_featurized,
    gc,
    get_memory_usage,
    lgb,
    max_ts,
    mo,
    pl,
    torch,
    xgb,
):
    mo.md("## 3. Machine Learning: Gradient Boosting (LGBM + XGB)")

    # Split back into Train/Valid/Test
    # Valid = last 10% of time indices in 'train' partition
    _split_cutoff = int(max_ts * 0.9)

    # Materialize masks lazily (we can't use boolean masks on LazyFrame easily for Numpy conversion, better to filter)

    print("Materializing Train Data...")
    # Filter and Select -> Collect -> Numpy. This creates a copy only of the needed data.
    _train_lf = full_df_featurized.filter(
        (pl.col("split") == "train") & (pl.col("ts_index") < _split_cutoff)
    )
    _X_train = _train_lf.select(feature_cols).collect().to_numpy()
    _y_train = _train_lf.select("y_target").collect().to_series().to_numpy()
    _w_train = _train_lf.select("weight").collect().to_series().to_numpy()

    print(f"Train Data Shape: {_X_train.shape}. Memory: {get_memory_usage():.2f} MB")

    print("Materializing Validation Data...")
    _valid_lf = full_df_featurized.filter(
        (pl.col("split") == "train") & (pl.col("ts_index") >= _split_cutoff)
    )
    _X_valid = _valid_lf.select(feature_cols).collect().to_numpy()
    _y_valid = _valid_lf.select("y_target").collect().to_series().to_numpy()
    _w_valid = _valid_lf.select("weight").collect().to_series().to_numpy()

    # Try to get horizon for evaluation breakdown
    try:
        _h_valid = _valid_lf.select("horizon").collect().to_series().to_numpy()
    except:
        _h_valid = None

    print("Materializing Test Data...")
    _test_lf = full_df_featurized.filter(pl.col("split") == "test")
    _X_test = _test_lf.select(feature_cols).collect().to_numpy()

    print(f"Total Memory after materialization: {get_memory_usage():.2f} MB")

    # Check for GPU availability
    # Check for GPU availability
    use_gpu = False
    try:
        print(f"Torch Version: {torch.__version__}")
        print(f"CUDA Version in Torch: {torch.version.cuda}")
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            use_gpu = True
            print(f"Selected GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Torch says CUDA is not available.")
    except Exception as e:
        print(f"Error checking GPU: {e}")

    print(f"GPU Available: {use_gpu}")

    # --- LGBM ---
    print(f"Training LightGBM ({'GPU' if use_gpu else 'CPU'})...")
    _lgb_params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "device": "gpu" if use_gpu else "cpu",
        "objective": "regression",
        "metric": "rmse",
        "verbose": -1,
    }

    _model_lgb = lgb.LGBMRegressor(**_lgb_params)
    _model_lgb.fit(
        _X_train,
        _y_train,
        sample_weight=_w_train,
        eval_set=[(_X_valid, _y_valid)],
        eval_sample_weight=[_w_valid],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    pred_lgb_val = _model_lgb.predict(_X_valid)
    pred_lgb_test = _model_lgb.predict(_X_test)
    print("LightGBM Done.")

    # --- Memory Cleanup (Optimized) ---
    # Delete model and force garbage collection
    try:
        del _model_lgb
    except:
        pass
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except:
        pass
    free_memory()

    # --- XGBoost ---
    print(f"Training XGBoost ({'GPU' if use_gpu else 'CPU'})...")

    # Low-Memory Configuration
    _xgb_params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "tree_method": "hist",
        "device": "cuda" if use_gpu else "cpu",
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "early_stopping_rounds": 50,
        "max_bin": 63,  # Reduces VRAM usage (~4x less histogram memory)
        "subsample": 0.8,  # Reduces peak memory
    }

    try:
        _model_xgb = xgb.XGBRegressor(**_xgb_params)
        _model_xgb.fit(
            _X_train,
            _y_train,
            sample_weight=_w_train,
            eval_set=[(_X_valid, _y_valid)],
            verbose=False,
        )
    except Exception as e:
        is_oom = "out of memory" in str(e).lower() or "allocate" in str(e).lower()
        if is_oom:
            print(
                "⚠️ XGBoost OOM with full data. Switching to Incremental (Batched) Training..."
            )

            # Tier 2: Aggressive Clean up + Batched GPU (Smaller Chunks)
            try:
                torch.cuda.empty_cache()
                gc.collect()

                _model_xgb = xgb.XGBRegressor(**_xgb_params)
                _n_chunks = 20  # increased from 5 to 20 to ensure it fits
                _chunk_size = len(_X_train) // _n_chunks
                _booster = None

                print(f"  Attempting {_n_chunks} batches (size ~{_chunk_size})...")

                for i in range(_n_chunks):
                    _start = i * _chunk_size
                    _end = (i + 1) * _chunk_size if i < _n_chunks - 1 else len(_X_train)

                    # Only validate on the last chunk to save VRAM overhead during training loops?
                    # No, we need it for consistency, but if it causes OOM we might need to skip.
                    # We'll try keeping it first.

                    _model_xgb.fit(
                        _X_train[_start:_end],
                        _y_train[_start:_end],
                        sample_weight=_w_train[_start:_end],
                        eval_set=[(_X_valid, _y_valid)],
                        xgb_model=_booster,
                        verbose=False,
                    )
                    _booster = _model_xgb.get_booster()

                    # Aggressive per-batch cleanup
                    gc.collect()
                    try:
                        torch.cuda.empty_cache()
                    except:
                        pass

            except Exception as batch_e:
                print(
                    f"⚠️ Batched GPU training failed ({batch_e}). Falling back to CPU..."
                )

                # Tier 3: CPU Fallback (Guaranteed to work, slower)
                _xgb_params["device"] = "cpu"
                # Ensure tree_method is compatible (hist is good for cpu too)
                _model_xgb = xgb.XGBRegressor(**_xgb_params)
                _model_xgb.fit(
                    _X_train,
                    _y_train,
                    sample_weight=_w_train,
                    eval_set=[(_X_valid, _y_valid)],
                    verbose=False,
                )
        else:
            raise e

    pred_xgb_val = _model_xgb.predict(_X_valid)
    pred_xgb_test = _model_xgb.predict(_X_test)
    print("XGBoost Done.")

    free_memory()
    free_memory()
    return pred_lgb_test, pred_lgb_val, pred_xgb_test, pred_xgb_val


@app.cell
def _(
    NBEATSModel,
    StatsForecastAutoARIMA,
    TimeSeries,
    free_memory,
    full_df_featurized,
    max_ts,
    mo,
    models_available,
    pl,
    stats_available,
    torch,
):
    mo.md("## 4. Deep Learning & Statistical Models (Darts Demo)")

    _split_cutoff = int(max_ts * 0.9)
    # Lazy filter expression
    _train_filter = (pl.col("split") == "train") & (pl.col("ts_index") < _split_cutoff)

    if models_available:
        print("Initializing Deep Learning Demo...")
        # Calculate top series using Lazy API
        _top_series_codes = (
            full_df_featurized.filter(_train_filter)
            .group_by("code")
            .agg(pl.col("weight").sum().alias("total_weight"))
            .sort("total_weight", descending=True)
            .limit(5)
            .select("code")
            .collect()  # Materialize to get the list
            .to_series()
            .to_list()
        )

        print(f"Selected Series for Demo: {_top_series_codes}")

        _train_list = []
        _val_list = []

        for _code in _top_series_codes:
            try:
                # Filter lazily then collect
                # Cast to Int64 to satisfy Darts' strict type checking
                # Aggregate by ts_index to ensure unique time steps (handle multiple sub_codes)
                _s_df = (
                    full_df_featurized.filter(pl.col("code") == _code)
                    .group_by("ts_index")
                    .agg(pl.col("y_target").mean())
                    .with_columns(pl.col("ts_index").cast(pl.Int64))
                    .sort("ts_index")
                    .collect()
                )
                _s_tr = _s_df.filter(pl.col("ts_index") < _split_cutoff)
                _s_va = _s_df.filter(pl.col("ts_index") >= _split_cutoff)

                if len(_s_tr) > 10 and len(_s_va) > 0:
                    _ts_tr = TimeSeries.from_dataframe(
                        _s_tr.to_pandas(),
                        time_col="ts_index",
                        value_cols="y_target",
                        fill_missing_dates=True,
                        freq=1,
                    )
                    _ts_va = TimeSeries.from_dataframe(
                        _s_va.to_pandas(),
                        time_col="ts_index",
                        value_cols="y_target",
                        fill_missing_dates=True,
                        freq=1,
                    )
                    _train_list.append(_ts_tr)
                    _val_list.append(_ts_va)
            except Exception as _e:
                print(f"Skipped {_code}: {_e}")

        if _train_list:
            # N-BEATS Demo
            print(f"Training N-BEATS Demo on {len(_train_list)} series...")
            _model_nbeats = NBEATSModel(
                input_chunk_length=30,
                output_chunk_length=7,
                n_epochs=5,
                batch_size=32,
                pl_trainer_kwargs={
                    "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
                    "enable_progress_bar": False,
                },
            )
            _model_nbeats.fit(_train_list, verbose=False)
            print("N-BEATS Demo Complete.")

            # StatsForecast Demo
            if stats_available:
                print("Running AutoARIMA Demo...")
                _model_arima = StatsForecastAutoARIMA()
                _model_arima.fit(_train_list[0])
                print("AutoARIMA Demo Complete.")
        else:
            print("Insufficient data for Darts demo.")
    else:
        print("Darts/Torch not available.")

    free_memory()
    return


@app.cell
def _(mo, np, pred_lgb_test, pred_lgb_val, pred_xgb_test, pred_xgb_val):
    mo.md("## 5. Ensembling & Submission")

    # Weighted Average Ensemble
    _ensemble_weights = [0.55, 0.45]

    final_pred_val = (
        _ensemble_weights[0] * pred_lgb_val + _ensemble_weights[1] * pred_xgb_val
    )

    final_pred_test = (
        _ensemble_weights[0] * pred_lgb_test + _ensemble_weights[1] * pred_xgb_test
    )

    # Clip predictions
    final_pred_test = np.maximum(final_pred_test, 0)

    print("Ensemble Predictions Generated.")
    return final_pred_test, final_pred_val


@app.cell
def _(final_pred_val, mo, np, _h_valid, _w_valid, _y_valid):
    mo.md("## 6. Local Evaluation (Validation Split)")

    def _clip01(x: float) -> float:
        return float(np.minimum(np.maximum(x, 0.0), 1.0))

    def weighted_rmse_score(y_target, y_pred, w) -> float:
        denom = np.sum(w * y_target**2)
        ratio = np.sum(w * (y_target - y_pred) ** 2) / denom
        clipped = _clip01(ratio)
        val = 1.0 - clipped
        return float(np.sqrt(val))

    print("-" * 60)
    print(f"{'METRIC':<20} | {'SCORE':<15} | {'ROWS':<10}")
    print("-" * 60)
    # Overall Score
    overall_score = weighted_rmse_score(_y_valid, final_pred_val, _w_valid)
    print(f"{'Local Skill Score':<20} | {overall_score:.6f}        | {len(_y_valid):,}")
    # By Horizon
    if _h_valid is not None:
        print("-" * 60)
        print("Scores by Horizon:")
        try:
            _horizons = np.unique(_h_valid)
            for h in sorted(_horizons):
                mask = _h_valid == h
                if mask.sum() > 0:
                    h_score = weighted_rmse_score(
                        _y_valid[mask], final_pred_val[mask], _w_valid[mask]
                    )
                    print(
                        f"  Horizon {h:<2}         | {h_score:.6f}        | {mask.sum():,}"
                    )
        except Exception as e:
            print(f"Could not calculate horizon breakdown: {e}")

    print("-" * 60)
    return


@app.cell
def _(final_pred_test, full_df_featurized, max_ts, mo, pl):
    mo.md("### Generate Submission CSV")

    _split_cutoff = int(max_ts * 0.9)
    # If working with LazyFrame upstream, we might need to handle _test_ids differently
    # But for submission, we can just use the materialized X_test indices if we tracked them,
    # Or just re-fetch IDs from LazyFrame (low cost)

    _test_ids = (
        full_df_featurized.filter(pl.col("split") == "test").select("id").collect()
    )

    submission_df = pl.DataFrame({"id": _test_ids["id"], "prediction": final_pred_test})

    submission_path = "submission_marimo.csv"

    # Check if submission_df is lazy
    if isinstance(submission_df, pl.LazyFrame):
        submission_df.collect().write_csv(submission_path)
    else:
        submission_df.write_csv(submission_path)

    print(f"Submission saved to: {submission_path}")
    mo.ui.table(submission_df.head(10))
    return


if __name__ == "__main__":
    app.run()
