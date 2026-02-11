import json
import os

notebook_cells = []

def add_cell(source_code, cell_type="code"):
    notebook_cells.append({
        "cell_type": cell_type,
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_code.splitlines(keepends=True)
    })

# --- Cell 1: Data Download ---
code_download = """
import os
import requests

def download_data(url, filepath):
    if os.path.exists(filepath):
        print(f"File {filepath} already exists. Skipping download.")
        return

    print(f"Downloading data from {url}...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded to {filepath}")
    else:
        print(f"Failed to download. Status code: {response.status_code}")

DATA_URL = "https://storage.googleapis.com/kagglesdsdata/competitions/105581/15271735/test.parquet?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1770992767&Signature=Fj%2F5LPgyVbzbph1WeG3uCJRMdqsabiq%2BKM3msHzs8y%2BXoHuAzoKKjvFDg3t8ueiPngilQtsFWg4FbUfzajf%2BDqzehOTGwRWzU6bX0xvdrrOeQusa7glpDDnF9n6C0izwzU8k%2FxFDdU7qI6vUtJLm3Yk20zfZMYx%2BuGtFrrtoTzcUx0k5ut%2Ft4OtyyBeCzpsUpCA5EjKMGbqRnB5P%2F7SCEWlwCQ7ZXL8w0kKJGCC3%2FYOqSktMDRhaeLsyP3lfCejur%2BxGfcd8hoLQWsEHOkYEw91k%2F%2BOy6vg5PkpYlQN2Exovmjg3o56VBIAZLLZhU%2BCAvtfL4X%2Bw02HrNJVKi5mhoA%3D%3D&response-content-disposition=attachment%3B+filename%3Dtest.parquet"
download_data(DATA_URL, "data/test.parquet")
"""
add_cell("# Step 0: Download Data", "markdown")
add_cell(code_download)

# --- Cell 2: Imports and Metric ---
code_imports = """
import polars as pl
import numpy as np
import warnings
import os
import lightgbm as lgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict
import requests

warnings.filterwarnings("ignore")

# Define the Weighted RMSE Score
def weighted_rmse_score(
    y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray
) -> float:

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    weights = np.asarray(weights)

    # Calculate weighted RMSE numerator and denominator
    weighted_squared_error = np.sum(weights * (y_true - y_pred) ** 2)
    weighted_y_squared = np.sum(weights * y_true**2)

    # Avoid division by zero
    if weighted_y_squared == 0:
        return 0.0

    # Calculate skill score
    score = 1 - np.sqrt(weighted_squared_error / weighted_y_squared)
    return score
"""
add_cell("# Imports & Metric Implementation", "markdown")
add_cell(code_imports)

# --- Cell 3: Iteration A (Data & Split) ---
code_iter_a = """
# Iteration A: The Golden Split & Metric (Polars)

def load_and_split_data(
    filepath: str = "data/test.parquet",
    target_col: str = "feature_ch",
    weight_col: str = "feature_cg",
    valid_ratio: float = 0.25,
) -> Tuple[pl.DataFrame, pl.DataFrame, list]:
    print("Loading data from", filepath)
    if not os.path.exists(filepath):
        # Create dummy data for testing if file doesn't exist
        print("Warning: Data file not found. Creating dummy data.")
        n_rows = 10000
        df = pl.DataFrame({
            "id": [f"c_sc_cat_h_{i}" for i in range(n_rows)],
            "ts_index": np.arange(n_rows),
            "code": np.random.choice(["A", "B"], n_rows).tolist(),
            "sub_code": np.random.choice(["X", "Y"], n_rows).tolist(),
            "sub_category": np.random.choice(["1", "2"], n_rows).tolist(),
            "horizon": np.random.choice([1, 10, 25], n_rows).tolist(),
            "feature_ch": np.random.randn(n_rows).tolist(),  # Target
            "feature_cg": np.random.uniform(0.5, 1.5, n_rows).tolist(), # Weight
        })
        # Add dummy features
        for i in range(10):
            df = df.with_columns(pl.lit(np.random.randn(n_rows)).alias(f"feature_{i}"))
    else:
        df = pl.read_parquet(filepath)
    
    print(f"Loaded {df.height:,} rows with {len(df.columns)} columns")

    # Determine split point based on ts_index
    min_ts = df["ts_index"].min()
    max_ts = df["ts_index"].max()
    ts_range = max_ts - min_ts
    split_ts = max_ts - int(ts_range * valid_ratio)

    print(f"Time index range: {min_ts} to {max_ts}")
    print(f"Validation split at ts_index >= {split_ts}")

    # Split data using filter (lazy or eager)
    train_df = df.filter(pl.col("ts_index") < split_ts)
    valid_df = df.filter(pl.col("ts_index") >= split_ts)
    
    # Feature columns (exclude meta)
    exclude_cols = ["id", "code", "sub_code", "sub_category", target_col, weight_col, "ts_index", "horizon"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    return train_df, valid_df, feature_cols

# Execute Iter A
train_df, valid_df, feature_cols = load_and_split_data()

# Baseline Calculation
y_true = valid_df["feature_ch"].to_numpy()
weights = valid_df["feature_cg"].fill_null(1.0).to_numpy() # Polars handles nulls differently
y_pred_baseline = np.ones_like(y_true) * train_df["feature_ch"].mean()
baseline_score = weighted_rmse_score(y_true, y_pred_baseline, weights)
print(f"Baseline (Mean Prediction) Score: {baseline_score:.4f}")
"""
add_cell("# Iteration A: Load Data & Baseline", "markdown")
add_cell(code_iter_a)

# --- Cell 3: Iteration B (Temporal Features - Polars) ---
code_iter_b = """
# Iteration B: Temporal Feature Engineering (Polars)
# Polars makes rolling windows extremely fast and clean.

def create_temporal_features_pl(
    df: pl.DataFrame,
    feature_cols: List[str],
    group_cols: List[str] = ["code", "sub_code", "sub_category"],
    rolling_windows: List[int] = [3, 7, 14, 30],
) -> pl.DataFrame:
    print("Creating temporal features with Polars...")
    
    # Ensure sorted by ts_index within groups
    df = df.sort(group_cols + ["ts_index"])
    
    features_to_process = feature_cols[:20] if len(feature_cols) > 50 else feature_cols
    
    # Define expressions for rolling/lag features
    exprs = []
    
    for feat in features_to_process:
        # Lags
        for lag in [1, 2, 3]:
            exprs.append(pl.col(feat).shift(lag).over(group_cols).alias(f"{feat}_lag_{lag}"))
            
        # Rolling Means (Shifted by 1 to prevent leakage)
        # Polars rolling operates on the column. .shift(1) ensures we don't peek.
        for window in rolling_windows:
            # rolling_mean usage: .rolling_mean(window_size).shift(1)
            # We must use .over(group_cols) to respect groups!
            # However, rolling_mean is deprecated in favor of rolling().mean()
            # To apply over groups efficiently:
            # We can use window functions inside over()
            
            # Note: naive .rolling() inside over() can be tricky in older Polars versions, 
            # but newer versions support it well.
            # Best practice: use creating rolling columns separately if over() is complex or use window functions.
            
            # Efficient pattern: 
            # (col(feat).shift(1).rolling_mean(window)).over(group_cols)
            # Shift FIRST to prevent leakage, then roll. Wait.
            # If we shift first, then rolling window at T includes T-1, T-2... which is safe.
            # Yes.
            
            col_name = f"{feat}_roll_{window}"
            exprs.append(
                pl.col(feat)
                .shift(1)
                .rolling_mean(window_size=window, min_periods=1)
                .over(group_cols)
                .alias(col_name)
            )
            
        # Expanding Mean (Shifted)
        # Cumulative sum / Count
        # Shift(1) first
        shifted = pl.col(feat).shift(1)
        exprs.append(
            (shifted.cum_sum() / shifted.cum_count())
            .over(group_cols)
            .alias(f"{feat}_exp_mean")
            .fill_nan(0) # Handle potential division by zero
        )

    # Apply all expressions at once! Ultra fast.
    df = df.with_columns(exprs)
    
    print(f"Created {len(exprs)} temporal features")
    return df

# Combine
full_df = pl.concat([train_df, valid_df])

# Create Features
full_df = create_temporal_features_pl(full_df, feature_cols)

# Re-split
# We need to calculate split_ts again or reuse
split_ts = full_df["ts_index"].max() - int((full_df["ts_index"].max() - full_df["ts_index"].min()) * 0.25)
train_df = full_df.filter(pl.col("ts_index") < split_ts)
valid_df = full_df.filter(pl.col("ts_index") >= split_ts)

# Update feature list
current_features = [c for c in full_df.columns if c not in ["id", "code", "sub_code", "sub_category", "feature_ch", "feature_cg", "ts_index", "horizon"]]
print(f"Total features after Iter B: {len(current_features)}")
"""
add_cell("# Iteration B: Temporal Features (Polars)", "markdown")
add_cell(code_iter_b)

# --- Cell 4: Iteration C (Weighted Model) ---
code_iter_c = """
# Iteration C: Weighted LightGBM (Polars -> Numpy)

def train_lgb_model(train_df, valid_df, features, params=None):
    if params is None:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "verbose": -1,
            "n_jobs": -1,
            "device": "gpu"
        }
    
    # Convert to numpy/pandas for LightGBM consumption (LightGBM supports Polars directly in newer versions too!)
    # But for safety and consistency with weights, we'll extract explicitly.
    
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

# Train separate models for horizons
horizons = sorted(train_df["horizon"].unique().to_list())
models = {}

print("Training Horizon-specific Models (Iter C)...")
for h in horizons:
    t_h = train_df.filter(pl.col("horizon") == h)
    v_h = valid_df.filter(pl.col("horizon") == h)
    
    if t_h.height == 0 or v_h.height == 0: continue
        
    model = train_lgb_model(t_h, v_h, current_features)
    models[h] = model
    
    # Score
    v_preds = model.predict(v_h.select(current_features).fill_null(0).to_numpy())
    score = weighted_rmse_score(v_h["feature_ch"].to_numpy(), v_preds, v_h["feature_cg"].fill_null(1.0).to_numpy())
    print(f"  Horizon {h} Score: {score:.4f}")

# Overall Evaluation C
# We can join predictions back
valid_df = valid_df.with_columns(pl.lit(0.0).alias("pred_iter_c"))

# Since we trained separate models, we iterate to update
# Polars update is immutable-ish. We can use map/when/then but loop is easier with to_pandas/numpy?
# Actually, let's keep predictions separate and join or update efficiently.
preds_full = []
for h, model in models.items():
    mask = valid_df["horizon"] == h
    # Filter
    sub_df = valid_df.filter(pl.col("horizon") == h)
    if sub_df.height > 0:
        preds = model.predict(sub_df.select(current_features).fill_null(0).to_numpy())
        # We need to map these back. 
        # Easier strategy: collect predictions with index/ID then join.
        temp_df = sub_df.select("id").with_columns(pl.Series(name="pred_iter_c_h", values=preds))
        preds_full.append(temp_df)

if preds_full:
    preds_all = pl.concat(preds_full)
    valid_df = valid_df.join(preds_all, on="id", how="left").with_columns(
        pl.col("pred_iter_c_h").fill_null(0).alias("pred_iter_c")
    )

overall_score_c = weighted_rmse_score(
    valid_df["feature_ch"].to_numpy(), 
    valid_df["pred_iter_c"].to_numpy(), 
    valid_df["feature_cg"].fill_null(1.0).to_numpy()
)
print(f"Overall Iteration C Score: {overall_score_c:.4f}")
"""
add_cell("# Iteration C: Weighted LightGBM", "markdown")
add_cell(code_iter_c)

# --- Cell 5: Iteration D (PCA Polars) ---
code_iter_d = """
# Iteration D: PCA (Polars -> Sklearn)

print("Applying PCA (Iter D)...")

# Select numeric features
pca_features = [c for c in train_df.columns if c.startswith("feature_") or "_roll_" in c]
pca_features = pca_features[:50]

# To Numpy for PCA
X_train_np = train_df.select(pca_features).fill_null(0).to_numpy()
X_valid_np = valid_df.select(pca_features).fill_null(0).to_numpy()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_np)
X_valid_scaled = scaler.transform(X_valid_np)

n_components=10
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_valid_pca = pca.transform(X_valid_scaled)

# Add back to Polars
# We create a dataframe of PCA feats and horizontal stack
pca_cols = [f"pca_{i}" for i in range(n_components)]
train_pca_df = pl.DataFrame(X_train_pca, schema=pca_cols)
valid_pca_df = pl.DataFrame(X_valid_pca, schema=pca_cols)

train_df = pl.concat([train_df, train_pca_df], how="horizontal")
valid_df = pl.concat([valid_df, valid_pca_df], how="horizontal")

features_d = current_features + pca_cols
# ... code for model training D (abbreviated, same pattern) ...
"""
add_cell("# Iteration D: PCA", "markdown")
add_cell(code_iter_d)

# --- Cell 6: Iteration E (Smoothed Target Encoding Polars) ---
code_iter_e = """
# Iteration E: Smoothed Target Encoding (Polars Fast)

def create_smoothed_target_encoding_pl(
    df, col, target="feature_ch", weight="feature_cg", min_samples=10, smoothing=10
):
    # Sort
    df = df.sort([col, "ts_index"])
    global_mean = df[target].mean()
    
    # Polars expressions for expanding mean
    # We want: (cumsum_shift * n_shift + smooth*global) / (n_shift + smooth)
    
    # Calculate Expanding Sum and Count
    # Use over()
    
    return df.with_columns(
        (
            (pl.col(target).shift(1).cum_sum().over(col).fill_null(0) + smoothing * global_mean) 
            / 
            (pl.col(target).shift(1).cum_count().over(col).fill_null(0) + smoothing)
        ).alias(f"{col}_enc_smooth")
    )

print("Applying Smoothed Target Encoding (Iter E)...")
full_df = pl.concat([train_df.select(pl.exclude(pca_cols)), valid_df.select(pl.exclude(pca_cols))])
# Need to re-gen PCA or just concat all? Reuse existing.
# Let's just concat all columns carefully.
# Actually, straightforward concat works if columns match.
full_df = pl.concat([train_df, valid_df])

for col in ["code", "sub_code", "sub_category"]:
    full_df = create_smoothed_target_encoding_pl(full_df, col)

# Re-split
train_df = full_df.filter(pl.col("ts_index") < split_ts)
valid_df = full_df.filter(pl.col("ts_index") >= split_ts)

features_e = features_d + [f"{c}_enc_smooth" for c in ["code", "sub_code", "sub_category"]]
"""
add_cell("# Iteration E: Smoothed Target Encoding", "markdown")
add_cell(code_iter_e)

# --- Cell 7: Iteration F (Interaction) ---
code_iter_f = """
# Iteration F: Interaction Features (Polars)

print("Creating Interaction Features (Iter F)...")
# Polars expression API makes this trivial
new_cols = []
base_feats = [c for c in features_e if "lag" in c][:5]

for feat in base_feats:
    new_cols.append((pl.col(feat) * pl.col("horizon")).alias(f"{feat}_x_horizon"))

new_cols.append((pl.col("horizon") ** 2).alias("horizon_squared"))

train_df = train_df.with_columns(new_cols)
valid_df = valid_df.with_columns(new_cols)

interaction_feats = [c.name for c in train_df.select(new_cols)] # Get names? alias sets name.
# Re-extract names
interaction_feats = [f"{feat}_x_horizon" for feat in base_feats] + ["horizon_squared"]

all_candidates_f = features_e + interaction_feats

# Feature Selection (Sklearn)
X_train_np = train_df.select(all_candidates_f).fill_null(0).to_numpy()
y_train_np = train_df["feature_ch"].to_numpy()

selector = SelectKBest(score_func=f_regression, k=min(100, len(all_candidates_f)))
selector.fit(X_train_np, y_train_np)
selected_indices = selector.get_support(indices=True)
selected_features_f = [all_candidates_f[i] for i in selected_indices]
print(f"Selected {len(selected_features_f)} features.")
"""
add_cell("# Iteration F: Feature Selection", "markdown")
add_cell(code_iter_f)

# --- Cell 8: Iteration G (Ensemble) ---
code_iter_g = """
# Iteration G: Ensemble

print("Training Ensemble (Iter G)...")

horizons = sorted(train_df["horizon"].unique().to_list())
param_sets = [
    {"num_leaves": 31, "learning_rate": 0.05, "bagging_fraction": 0.8},
    {"num_leaves": 63, "learning_rate": 0.03, "bagging_fraction": 0.9},
]

preds_all_g = []

for h in horizons:
    t_h = train_df.filter(pl.col("horizon") == h)
    v_h = valid_df.filter(pl.col("horizon") == h)
    if t_h.height == 0: continue
    
    horizon_preds = []
    for p in param_sets:
        full_params = {"objective": "regression", "metric": "rmse", "verbose": -1, "n_jobs": -1, "device": "gpu", **p}
        model = train_lgb_model(t_h, v_h, selected_features_f, params=full_params)
        preds = model.predict(v_h.select(selected_features_f).fill_null(0).to_numpy())
        horizon_preds.append(preds)
    
    avg_preds = np.mean(horizon_preds, axis=0)
    
    # Store with ID
    temp_df = v_h.select("id").with_columns(pl.Series("prediction", avg_preds))
    preds_all_g.append(temp_df)

if preds_all_g:
    submission = pl.concat(preds_all_g)
    submission.write_csv("submission_final_polars.csv")
    print("Saved submission_final_polars.csv")
"""
add_cell("# Iteration G: Ensemble", "markdown")
add_cell(code_iter_g)

# Write
notebook = {
    "cells": notebook_cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.5"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("solution.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)

print("Generated solution.ipynb with Polars")
