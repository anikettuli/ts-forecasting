# AGENTS.md - Development Guidelines for Agentic Coding

Guidelines for agentic coding systems in `ts-forecasting` repository.

## Project Overview

**Type**: Python GPU-accelerated time series forecasting competition
**Python Version**: 3.13.12
**Hardware**: RTX 3070 + CUDA 13.1 + CuPy 13.x
**Environment**: Virtual environment at `.venv/` (Windows-style, uses Scripts/)
**Primary Libraries**: cupy, polars, lightgbm, xgboost, catboost, scikit-learn
**Data Format**: Parquet files in `data/` directory
**Main Notebook**: `solution.ipynb` (contains all training code)
**Evaluator**: `evaluate.py` (local submission scoring)

**Target Column**: `y_target` (continuous values, NOT feature_ch)
**Weight Column**: `weight` (NOT feature_cg - extreme skew 0 to 13.9T)

## Build, Test & Lint Commands

```bash
# Activate venv (Windows)
.venv\Scripts\activate

# Run Jupyter notebook (main development)
.venv/Scripts/jupyter.exe notebook solution.ipynb

# Quick GPU test
.venv/Scripts/python.exe -c "import cupy as np; print(f'GPU: {np.cuda.runtime.getDeviceCount()} devices')"

# Evaluate a submission locally
.venv/Scripts/python.exe evaluate.py submission_optimized.csv

# Setup from scratch
python3.13 -m venv .venv
.venv/Scripts/pip.exe install -r requirements.txt
```

## Code Style Guidelines

### 1. Imports & Formatting

```python
# Order: stdlib -> third-party -> local
import gc
import os
import warnings
from typing import List, Tuple, Dict

import cupy as np          # GPU operations
import numpy as np_cpu     # CPU operations for sklearn/external libs
import polars as pl
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
```

- Max line: 100 chars, 4-space indent
- CuPy imported as `np`, NumPy as `np_cpu`

### 2. GPU/CPU Conversion Pattern

**Critical**: Matrix operations -> CuPy (GPU). External libs -> NumPy (CPU).

```python
def gpu_to_cpu(x):
    """CuPy GPU -> NumPy CPU (handles scalars + arrays)."""
    if x is None:
        return None
    try:
        if isinstance(x, (float, int, np_cpu.generic)):
            return x
        return x.get() if hasattr(x, 'get') else np_cpu.asarray(x)
    except Exception:
        return np_cpu.asarray(x)

def cpu_to_gpu(x):
    """NumPy CPU -> CuPy GPU."""
    return np.asarray(x) if x is not None else None
```

### 3. Type Hints

```python
def weighted_rmse_score(
    y_target: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray
) -> float:
    """GPU-accelerated weighted RMSE skill score."""
```

### 4. Naming Conventions

- `snake_case` for functions/variables
- `PascalCase` for classes
- `UPPER_SNAKE_CASE` for constants

### 5. Error Handling

- Division by zero: `np.where(std == 0, 1.0, std)` or add `+ 1e-8`
- GPU scalars: `float(gpu_scalar)` before passing to sklearn
- Weight clipping: `np.clip(weights, 0, np.percentile(weights, 99.9))`

## Data Pipeline Patterns

### Polars Temporal Features (Leakage-Safe)

```python
# CRITICAL: Sort before grouped operations
df = df.sort(["code", "sub_code", "ts_index"])

# Lag (strictly causal)
pl.col(feat).shift(1).over(group_cols).alias(f"{feat}_lag1")

# Rolling mean with shift
pl.col(feat).shift(1).rolling_mean(window_size=7, min_periods=1).over(group_cols)

# Expanding mean
(pl.col(feat).shift(1).cum_sum().over(group_cols) /
 (pl.col(feat).shift(1).cum_count().over(group_cols) + 1e-8))

# Rate of change
((pl.col(feat) - pl.col(feat).shift(1).over(group_cols)) /
 (pl.col(feat).shift(1).over(group_cols).abs() + 1e-8))
```

### Weight Handling

```python
# Log-transform weights (handles extreme skew: 0 to 13.9 trillion)
w_raw = df["weight"].fill_null(1.0).to_numpy()
w_transformed = np_cpu.log1p(w_raw) * time_decay
```

## Key Functions

| Function | Purpose |
|----------|---------|
| `weighted_rmse_score()` | Skill Score = sqrt(1 - sum(w*(y-y_hat)^2) / sum(w*y^2)) |
| `fast_eval()` | Quick LGBM evaluation for iteration tracking |
| `create_temporal_features_single()` | Lags, rolling, expanding, rate of change |
| `train_horizon_model()` | Horizon-specific model training |

## Common Issues to Avoid

- **Wrong target**: Use `y_target`, NOT `feature_ch`
- **Wrong weight**: Use `weight`, NOT `feature_cg`
- **Temporal leakage**: Always `.shift(1)` for features
- **Bad validation split**: Use temporal split (last 10% of train period)
- **Missing weights**: `.fill_null(1.0)` (Polars)
- **Cold-start**: Handle new `sub_codes` in test (35 entities)

## Submission Format

```csv
id,prediction
W2MW3G2L__495MGHFJ__PZ9S1Z4V__3__3647,0.123
...
```

- Must match all test IDs exactly
- Join back to original test order to ensure completeness

## Quick Checklist

- [ ] Target is `y_target` (not `feature_ch`)
- [ ] Weight is `weight` (not `feature_cg`)
- [ ] Temporal features use `.shift(1)` for leakage prevention
- [ ] Validation split is temporal (last 10% of train period)
- [ ] Weights log-transformed for extreme skew
- [ ] GPU operations with CuPy, CPU ops with NumPy
- [ ] All test IDs included in submission
