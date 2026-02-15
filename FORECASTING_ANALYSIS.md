# Forecasting Methodology Analysis

## Current Performance
- **Internal Validation Score:** 0.1711
- **Official Kaggle Score:** 0.2101
- **Target:** > 0.25

## Why Internal Score Underestimates Official

1. **Validation Weight Distribution Mismatch**
   - Train weights median: 1,699
   - Validation (last 10%) weights median: 374
   - Test weights: unknown, but different from validation

2. **Cold-Start Entities**
   - 74.5% of test sub_codes are NOT in training data
   - Model handles cold-start better than validation suggests

3. **Temporal Distribution**
   - Train: ts_index [1, 3601]
   - Test: ts_index [3602, 4376]
   - Test period may have "calmer" target values

## Methodology Strengths
- Per-horizon models (correct for different forecast horizons)
- LightGBM with categorical features
- Historical aggregates as baseline
- Ensemble optimization on validation

## Methodology Weaknesses
- Single validation split (high variance)
- No temporal features (lags, rolling stats)
- Suboptimal cold-start handling
- No cross-validation

## Recommendations for Further Improvement

### Priority 1: Cold-Start Handling
Create hierarchical fallback: sub_code → code → sub_category → global_mean

### Priority 2: Validation Strategy
- Use ensemble of multiple temporal splits (5%, 10%, 15%)
- Create cold-start specific validation subset

### Priority 3: Feature Engineering
- Add temporal features (lags, rolling means) on train only
- Target encoding with smoothing
- Time-based features (day of week sin/cos)

### Priority 4: Model Improvements
- Try CatBoost for categorical handling
- Use log-transformed weights for training
- Optimize per-horizon hyperparameters

## Files
- `forecasting.py` - Main marimo notebook
- `plot_submission.py` - Visualization notebook
- `submission_optimized.csv` - Best submission (score: 0.2101)
