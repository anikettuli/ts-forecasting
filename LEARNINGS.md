# Time Series Forecasting - Learnings & Cheat Sheet

## Score Progression

| Version | Score | Change |
|---------|-------|--------|
| Baseline (v6) | 0.1592 | Starting point |
| v8 | 0.1671 | +0.0079 - Best achieved |
| v13 (log weights) | 0.1652 | -0.0019 - Regression |
| v14 (raw weights + 800 rounds) | 0.1561 | -0.0110 - Major regression |
| v11 (diverse configs) | 0.1628 | -0.0043 - Regression |

---

## What Worked (+Score)

### 1. Multi-Seed LGBM Ensemble
- **Seeds**: 42, 123, 456
- **Leaves**: 63, 127, 31 (diverse complexity)
- **Learning Rates**: 0.03, 0.02, 0.04 (diverse speeds)
- **Result**: Averaging predictions from diverse configs improves stability

### 2. Historical Mean Blend
- **Feature**: `hist_mean = mean(y_target) GROUP BY code, sub_category, horizon`
- **Blend Ratio**: Alpha search 0.5-1.0 per horizon
- **Horizon-specific**: Each horizon has optimal blend weight
- **Result**: Short horizons benefit more from blend (H1: 0.95, H25: 1.0)

### 3. Target Encoding
- **code_te**: Mean y_target by code
- **sub_code_te**: Mean y_target by sub_code
- **Result**: Captures entity-level patterns

### 4. Cyclical Time Features
- `dow_sin`, `dow_cos` (7-day cycle)
- **Result**: Minor but consistent improvement

### 5. Cold-Start Indicator
- Binary flag for new sub_codes in test
- **Result**: Helps model identify entities with no history

---

## What Regressed (-Score)

### 1. Raw Weights (No Clipping) → 0.1561 (-0.011)
- Using unclipped weights caused extreme outliers to dominate
- **Fix**: Always clip weights at 99.9 percentile

### 2. Log Weights → 0.1652 (-0.002)
- `log1p(weight)` for training reduced score
- Original clipped weights work better

### 3. More Trees (800 rounds) → 0.1561 (-0.011)
- More rounds with early stopping didn't help
- 500 rounds with early_stopping(30) is optimal

### 4. XGB + CAT Ensemble
- Timed out frequently
- When it ran, didn't improve over LGBM-only
- **Insight**: LGBM ensemble sufficient for this problem

### 5. More Seeds (5 seeds)
- Timed out
- 3 seeds with diverse configs is the sweet spot

---

## Optimal Configuration (v8 - 0.1671)

```python
# Model Configs
configs = [
    {"seed": 42, "leaves": 63, "lr": 0.03},
    {"seed": 123, "leaves": 127, "lr": 0.02},
    {"seed": 456, "leaves": 31, "lr": 0.04},
]

# Training
num_boost_round = 500
early_stopping = 30
feature_fraction = 0.8
bagging_fraction = 0.8

# Weights
w_tr = np.clip(weight, 0, np.percentile(weight, 99.9))

# Features
feature_cols = raw_features + [
    "horizon", "code_te", "sub_code_te", 
    "hist_mean", "hist_std", "is_cold", 
    "dow_sin", "dow_cos"
]

# Blend optimization per horizon
for alpha in np.arange(0.5, 1.01, 0.05):
    pred = alpha * lgb_pred + (1 - alpha) * hist_mean
```

---

## Horizon-Specific Patterns

| Horizon | LGB Score | Hist Score | Best Blend | Notes |
|---------|-----------|------------|------------|-------|
| 1 | 0.059 | 0.039 | 0.95 | Short-term, LGB dominates |
| 3 | 0.086 | 0.058 | 1.00 | LGB only |
| 10 | 0.152 | 0.110 | 0.90 | Medium-term, some blend |
| 25 | 0.195 | 0.091 | 1.00 | Long-term, LGB only |

**Insight**: Longer horizons harder (lower score), but LGB consistently beats historical mean.

---

## Failed Approaches

| Approach | Result | Reason |
|----------|--------|--------|
| Temporal features (lags, rolling) | Memory error | Too many features on combined train+test |
| XGB + CAT ensemble | Timeout | 3 models per horizon too slow |
| Log weight transform | -0.002 | Compresses weight variance too much |
| 5 seeds | Timeout | Diminishing returns after 3 |
| 800 boosting rounds | -0.011 | Overfitting without proper early stopping |

---

## Next Steps to Try

1. **Feature Selection**: Use LGBM importance to select top 50 features, reduce memory
2. **Different Validation Split**: Try 80/20 instead of 90/10
3. **Post-processing**: Clip predictions to training range (already doing)
4. **Hierarchical Features**: Add category-level aggregations
5. **Time Decay**: Weight recent data more heavily in training

---

## Quick Commands

```bash
# Run solution
.venv/Scripts/python.exe solution.py

# Evaluate submission
.venv/Scripts/python.exe evaluate.py submission_optimized.csv

# Commit score improvement
git add solution.py submission_optimized.csv && git commit -m "score: X.XXXX - description"
```
