# AGENTS.md - Development Guidelines for Agentic Coding

This document provides guidelines for agentic coding systems operating in the `ts-forecasting` repository.

## Project Overview

**Type**: Python time series forecasting project  
**Python Version**: 3.13.7  
**Environment**: Virtual environment at `.venv/`  
**Key Dependencies**: pandas, numpy, lightgbm, scikit-learn, xgboost

## Build, Test & Lint Commands

### Running Scripts

```bash
# Activate virtual environment
source .venv/bin/activate

# Run any single script
.venv/bin/python iteration_a.py
.venv/bin/python iteration_b.py
.venv/bin/python run_pipeline.py

# Run full pipeline
.venv/bin/python run_pipeline.py
```

### Running Single Tests

There is no formal test suite yet. To validate changes:

```bash
# Test a single module by running it
.venv/bin/python <module_name>.py

# Import and test manually
.venv/bin/python -c "from iteration_a import weighted_rmse_score; import numpy as np; print(weighted_rmse_score(np.array([1,2,3]), np.array([1,2,3]), np.array([1,1,1])))"

# Validate code with ruff (if needed)
.venv/bin/python -m ruff check <file.py>
```

### Environment Setup

```bash
# If venv doesn't exist
python3.13 -m venv .venv

# Install dependencies
.venv/bin/pip install pandas numpy lightgbm scikit-learn xgboost

# Check versions
.venv/bin/pip list
```

---

## Code Style Guidelines

### 1. Imports

**Rules:**
- Group imports in this order: stdlib → third-party → local
- Each group separated by a blank line
- Sort alphabetically within groups
- Use explicit imports, not `import *`
- Type hints require `from typing import ...`

**Good:**
```python
import json
import warnings
from typing import Tuple, List, Dict

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

import iteration_a
from features import create_all_features
```

**Bad:**
```python
from features import *
import pandas as pd, numpy as np
from typing import *
```

---

### 2. Formatting

**Rules:**
- Max line length: 100 characters (implicit from codebase)
- Use 4 spaces for indentation (never tabs)
- Two blank lines between top-level functions/classes
- One blank line between methods
- No trailing whitespace

**Function spacing:**
```python
def function_one():
    """Docstring."""
    pass


def function_two():
    """Docstring."""
    pass
```

---

### 3. Type Hints

**Rules:**
- Always use type hints for function parameters and return types
- Use `np.ndarray` for numpy arrays, `pd.DataFrame` for dataframes
- Use `Optional[T]` for nullable values
- Use tuple unpacking annotations: `Tuple[str, int, list]`

**Good:**
```python
def weighted_rmse_score(
    y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray
) -> float:
    """Calculate weighted RMSE skill score."""
    pass

def load_and_split_data(
    filepath: str = "data/test.parquet",
    valid_ratio: float = 0.25,
) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    """Load and split data."""
    pass
```

**Bad:**
```python
def weighted_rmse_score(y_true, y_pred, weights):
    pass
```

---

### 4. Naming Conventions

**Rules:**
- `snake_case` for functions, variables, module names
- `PascalCase` for classes (rarely used here)
- `UPPER_SNAKE_CASE` for constants
- Descriptive names; avoid single letters except `i`, `j` in loops
- Prefix private/internal vars with underscore: `_group_key`, `_temp`

**Good:**
```python
def create_temporal_features(df, base_features):
    pass

feature_cols = [c for c in df.columns if c.startswith("feature_")]
MAX_HORIZON = 100
```

**Bad:**
```python
def CreateTemporalFeatures(df, bf):
    pass

f = []
```

---

### 5. Docstrings

**Rules:**
- All public functions require docstrings
- Use Google-style docstrings (summary, args, returns)
- One-line summary ends with period
- Add examples for complex functions

**Good:**
```python
def weighted_rmse_score(
    y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray
) -> float:
    """
    Calculate weighted RMSE skill score.

    SkillScore = 1 - sqrt(sum(w * (y - y_hat)^2) / sum(w * y^2))

    Args:
        y_true: Ground truth target values
        y_pred: Predicted values
        weights: Sample weights

    Returns:
        Skill score (higher is better, 1.0 is perfect)
    """
```

---

### 6. Error Handling

**Rules:**
- Avoid bare `except:` clauses
- Catch specific exceptions
- Use meaningful error messages
- Check for division by zero, NaN, infinity

**Good:**
```python
if weighted_y_squared == 0:
    return 0.0

try:
    df = pd.read_parquet(filepath)
except FileNotFoundError as e:
    print(f"Data file not found: {filepath}")
    raise

if df.isna().any():
    print("Warning: NaN values detected")
```

**Bad:**
```python
try:
    result = sum(array) / 0
except:
    pass
```

---

### 7. Data Pipeline Patterns

**Rules:**
- Always use `.copy()` when creating new dataframes from existing ones
- Sort dataframes before grouping for consistent results
- Prevent temporal leakage: use `.shift(1)` before features
- Reset index after major transformations: `.reset_index(drop=True)`
- Use group keys for groupby operations

**Good:**
```python
df = df.copy()
df = df.sort_values(["code", "sub_code", "ts_index"]).reset_index(drop=True)

for lag in [1, 2, 3]:
    df[f"feature_lag{lag}"] = df.groupby("code")[feature].shift(lag)

df["rolling_mean"] = df.groupby("code")[feature].transform(
    lambda x: x.rolling(7, min_periods=1).mean().shift(1)
)
```

---

### 8. Common Issues to Avoid

- **Temporal leakage**: Always shift features by 1 when using future data for feature engineering
- **Missing weights**: Use `.fillna(1.0)` for weight columns
- **Index confusion**: Reset index after major groupby operations
- **Mutating inputs**: Always start with `df = df.copy()`
- **Division by zero**: Check denominator before division
- **Unmatched dtypes**: Ensure arrays are `np.asarray()` before math operations

---

## File Organization

```
ts-forecasting/
├── iteration_a.py           # Baseline: metric & data loading
├── iteration_b.py           # Feature engineering v1
├── iteration_c.py           # LightGBM baseline model
├── iteration_d.py           # Enhanced features + PCA
├── iter_e_smoothed_encoding.py
├── features.py              # Feature creation functions
├── model.py                 # Model training utilities
├── model_optimized.py       # Optimized model variant
├── metrics.py               # Metric calculations
├── run_pipeline.py          # Full pipeline script
├── full_solution.py         # Complete end-to-end solution
├── data/                    # Data directory (test.parquet)
└── .venv/                   # Python virtual environment
```

---

## Key Functions & Utilities

- `weighted_rmse_score()`: Custom scoring metric (implementation varies per file)
- `load_and_split_data()`: Data loading with train/validation split
- `create_all_features()`: Comprehensive feature engineering pipeline
- `create_temporal_features()`: Lag and rolling window features
- `create_target_encoding()`: Target encoding with leakage prevention

---

## Quick Checklist Before Committing

- [ ] Functions have type hints (params and return)
- [ ] Public functions have docstrings
- [ ] Imports are organized and sorted
- [ ] Lines under 100 characters
- [ ] `.copy()` used for dataframe mutations
- [ ] No temporal leakage in feature engineering
- [ ] Error handling for edge cases (NaN, zero division)
- [ ] Code follows `snake_case` naming
- [ ] Tested by running the script end-to-end
