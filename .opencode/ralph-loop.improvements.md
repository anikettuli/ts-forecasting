---
active: true
iteration: 0
maxIterations: 50
---

# Kaggle TS-Forecasting: Improve Score from 0.2101 to > 0.25

## Context
- Competition: https://www.kaggle.com/competitions/ts-forecasting/overview
- Current Official Score: 0.2101
- Target Score: > 0.25
- Main Files: forecasting.py (marimo notebook), solution.py (standalone script)
- Data: data/train.parquet, data/test.parquet
- Target Column: y_target
- Weight Column: weight (extreme skew: 0 to 13.9 trillion, clip at 99.9 percentile)

## Critical Insights from Analysis

### Cold-Start Problem (CRITICAL)
- **74.5% of test sub_codes are NOT in training data** (only 12 of 47 overlap)
- Current solution uses historical aggregates which return global_mean for cold-start
- This is a MAJOR opportunity for improvement

### Validation-Test Mismatch
- Internal validation score (0.1711) underestimates official (0.2101)
- Validation weights (median=374) differ from train weights (median=1,699)
- Test period (ts_index 3602-4376) is strictly after training period (1-3601)

### Score Metric
- weighted_rmse_score = sqrt(1 - SSE/SST)
- SSE = sum(w * (y - y_pred)^2), SST = sum(w * y^2)
- High-weight samples have y close to 0, small errors matter hugely
- Need to reduce SSE by ~20% to reach 0.25

## Git Workflow

```bash
# Create new branch for improvements
git checkout -b score-improvements

# After each successful improvement:
git add solution.py submission_optimized.csv
git commit -m "score: X.XXXX - brief description of improvement"
```

## Improvement Strategy (Ordered by Priority)

### Phase 1: Cold-Start Handling (Expected: +0.02-0.04)
1. Implement hierarchical fallback encoding:
   - Level 1: code + sub_code + horizon (most specific)
   - Level 2: code + sub_category + horizon
   - Level 3: code + horizon
   - Level 4: sub_category + horizon
   - Level 5: horizon only
   - Level 6: global_mean (fallback)

2. Create cold-start indicator feature
3. Weight encodings by sample count (smoothing)
4. Test each level individually to find optimal fallback

### Phase 2: Temporal Features (Expected: +0.01-0.03)
1. Add lag features computed on TRAIN ONLY:
   - y_lag1, y_lag7, y_lag14, y_lag28
   - For test data, fill with historical aggregate values

2. Add rolling statistics:
   - rolling_mean_7, rolling_mean_28
   - rolling_std_7

3. Add time-based features:
   - dow_sin, dow_cos (day of week)
   - dom_sin, dom_cos (day of month)
   - Momentum features

### Phase 3: Model Improvements (Expected: +0.01-0.02)
1. Try different weight transformations:
   - Raw clipped (current)
   - log1p transformed
   - sqrt transformed
   - Rank-based

2. Add CatBoost for categorical handling
3. Per-horizon hyperparameter tuning
4. More diverse ensemble (different seeds, architectures)

### Phase 4: Validation Strategy (Expected: better estimation)
1. Implement multiple temporal splits:
   - Last 5%, 10%, 15% of training data
   - Report average and std of scores

2. Create cold-start validation subset:
   - Hold out some sub_codes from training
   - Evaluate specifically on cold-start performance

## Execution Workflow (Repeat Until Complete)

### Step 1: Run Current Solution
```bash
.venv/Scripts/python.exe solution.py
```
Extract the validation score from output.

### Step 2: Make One Targeted Improvement
Choose the next improvement from the priority list above.
- Make focused changes to solution.py
- Test the change locally
- Verify no errors

### Step 3: Evaluate Improvement
```bash
.venv/Scripts/python.exe solution.py
```
Compare new validation score to previous.

### Step 4: Generate Submission
If validation score improved:
```bash
cp submission_optimized.csv submission_v[X].csv
```

### Step 5: Commit
```bash
git add solution.py submission_optimized.csv
git commit -m "score: X.XXXX - description of improvement"
```

### Step 6: Repeat
Continue to next improvement until official score > 0.25.

## Specific Code Changes to Try

### Change 1: Hierarchical Encoding
```python
# Add to solution.py after loading data
def create_hierarchical_encodings(tr):
    encodings = {}
    global_mean = tr["y_target"].mean()
    
    # Level 1: Most specific
    encodings['l1'] = tr.group_by(["code", "sub_code", "horizon"]).agg([
        pl.col("y_target").mean().alias("pred"),
        pl.col("y_target").count().alias("cnt"),
    ])
    
    # Level 2-5: Less specific
    # ... implement other levels
    
    return encodings, global_mean

def get_hierarchical_pred(row, encodings, global_mean):
    # Try each level, return first match
    for level in ['l1', 'l2', 'l3', 'l4', 'l5']:
        # Check if key exists
        # Return prediction if found
    return global_mean
```

### Change 2: Weight Transformation
```python
# Try in training loop
w_tr_transformed = np.log1p(np.clip(w_tr, 0, np.percentile(w_tr, 99.9)))
```

### Change 3: Cold-Start Feature
```python
# Add feature indicating cold-start
df = df.with_columns(
    pl.when(pl.col("sub_code").is_in(train_sub_codes)).then(0).otherwise(1).alias("is_cold_start")
)
```

## Success Criteria
- [ ] Official Kaggle score > 0.25
- [ ] Each improvement committed with score in message
- [ ] Validation methodology documented
- [ ] All code runs without errors

When score > 0.25 is confirmed, output: <promise>DONE</promise>
