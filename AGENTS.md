# AGENTS.md - Development Guidelines for Agentic Coding

Guidelines for agentic coding systems in `ts-forecasting` repository.

## Project Overview

**Type**: Python GPU-accelerated time series forecasting  
**Python Version**: 3.13.12  
**Hardware**: RTX 3070 + CUDA 13.1 + CuPy 13.x  
**Environment**: Virtual environment at `.venv/` (Windows-style, uses Scripts/)  
**Primary Libraries**: cupy, polars, lightgbm, scikit-learn  
**Data Format**: Parquet files in `data/` directory  
**Main Notebook**: `solution.ipynb` (contains all code)

## Build, Test & Lint Commands

```bash
# Activate venv (WSL style)
source .venv/Scripts/activate

# Run Jupyter notebook
.venv/Scripts/jupyter.exe notebook solution.ipynb

# Quick GPU test
.venv/Scripts/python.exe -c "import cupy as np; print(f'GPU: {np.cuda.runtime.getDeviceCount()} devices')"

# Setup
python3.13 -m venv .venv
.venv/Scripts/pip.exe install -r requirements.txt
```

---

## Code Style Guidelines

### 1. GPU/CPU Conversion Pattern

**Critical**: Matrix operations → CuPy (GPU). External libs → NumPy (CPU).

```python
import cupy as np
import numpy as np_cpu

def gpu_to_cpu(x):
    """CuPy GPU → NumPy CPU (handles scalars + arrays)."""
    if x is None:
        return None
    try:
        if hasattr(x, 'get'):
            return x.get()
        elif hasattr(x, 'item'):
            return x.item()
        else:
            return np_cpu.asarray(x)
    except Exception as e:
        return np_cpu.asarray(x)

def cpu_to_gpu(x):
    """NumPy CPU → CuPy GPU."""
    return np.asarray(x) if x is not None else None
```

### 2. Imports & Formatting
- Group: stdlib → third-party → local
- CuPy imported as `np`, NumPy as `np_cpu`
- Max line: 100 chars, 4-space indent

```python
import os
import sys
from typing import Tuple, List

import cupy as np
import numpy as np_cpu
import polars as pl
import lightgbm as lgb
from sklearn.decomposition import PCA
```

### 3. Type Hints
- All params and returns typed
- Use `np.ndarray` for GPU arrays

```python
def weighted_rmse_score(
    y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray
) -> float:
    """GPU-accelerated metric. Returns Python float."""
```

### 4. Naming Conventions
- `snake_case` for functions/variables
- `PascalCase` for classes
- `UPPER_SNAKE_CASE` for constants

### 5. Error Handling
- Check division by zero on GPU: `np.where(std == 0, 1.0, std)`
- GPU scalars must be converted: `float(train_mean_gpu)`

---

## Data Pipeline Patterns

### Polars (Primary) + CuPy
```python
# Sort before grouped operations
df = df.sort(["code", "sub_code", "ts_index"])

# Lags with leakage prevention (Polars)
pl.col(feat).shift(1).over(group_cols).alias(f"{feat}_lag1")

# Rolling with shift (Polars)
pl.col(feat).shift(1).rolling_mean(window_size=7, min_periods=1).over(group_cols)

# Expanding mean with shift (Polars)
shifted = pl.col(feat).shift(1)
(shifted.cum_sum() / shifted.cum_count()).over(group_cols).alias(f"{feat}_exp").fill_nan(0)

# GPU-accelerated metric
y_true_gpu = cpu_to_gpu(df["feature_ch"].to_numpy())
weights_gpu = cpu_to_gpu(df["feature_cg"].fill_null(1.0).to_numpy())
score = float(weighted_rmse_score(y_true_gpu, y_pred_gpu, weights_gpu))
```

### GPU Acceleration Pattern
```python
# Load to GPU, process, convert to CPU once
X_train_gpu = cpu_to_gpu(train_df.select(feature_cols).fill_null(0).to_numpy())
mean_gpu = np.mean(X_train_gpu, axis=0, keepdims=True)
X_train_scaled = (X_train_gpu - mean_gpu) / std_gpu

# Convert to CPU for sklearn/LightGBM
X_train_np = gpu_to_cpu(X_train_scaled)
```

---

## Common Issues to Avoid

- **Temporal leakage**: Always `.shift(1)` for features
- **Missing weights**: `.fill_null(1.0)` (Polars)
- **GPU scalar conversion**: `float(gpu_scalar)` before passing to sklearn
- **Division by zero**: `np.where(std == 0, 1.0, std)` on GPU
- **Group confusion**: `.sort()` before `.over()` operations
- **Excessive GPU↔CPU transfers**: Batch operations, convert once at boundaries

---

## File Organization

```
ts-forecasting/
├── solution.ipynb            # Main notebook with 7-step pipeline
├── data/test.parquet          # Main dataset
├── requirements.txt           # Dependencies
└── .venv/Scripts/            # Python virtual environment
```

---

## Key Functions

- `weighted_rmse_score()`: SkillScore = 1 - sqrt(sum(w*(y-y_hat)²)/sum(w*y²))
- `load_and_split_data()`: Split by ts_index
- `gpu_to_cpu()` / `cpu_to_gpu()`: GPU↔CPU conversion
- `create_temporal_features_pl()`: Lags, rolling, expanding (Polars)
- `create_smoothed_target_encoding_pl()`: Target encoding with `.shift(1)`

---

## Model Training

```python
# GPU preprocessing, CPU LightGBM
X_train_gpu = cpu_to_gpu(train_df.select(features).fill_null(0).to_numpy())
X_train_np = gpu_to_cpu(X_train_gpu)

train_data = lgb.Dataset(X_train_np, label=y_train_np, weight=w_train_np)

params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "verbose": -1,
    "n_jobs": -1,
    "device": "gpu"
}

model = lgb.train(params, train_data, num_boost_round=1000,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(False)])
```

---

## Quick Checklist

- [ ] GPU operations with CuPy, CPU ops with NumPy
- [ ] Type hints on functions
- [ ] Lines under 100 chars
- [ ] `.shift(1)` for temporal features
- [ ] Null handling with `.fill_null()`
- [ ] GPU scalars converted to Python floats
- [ ] Tested in Jupyter notebook