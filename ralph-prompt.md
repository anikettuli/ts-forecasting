# Kaggle TS-Forecasting Competition - Ralph Loop Task

## Goal
Improve the time series forecasting model to achieve an internal performance metric above 25% (0.25) and generate sensible predictions.

## Context
- Competition: https://www.kaggle.com/competitions/ts-forecasting/overview
- Main file to improve: `forecasting.py` (a marimo notebook)
- Visualization: `plot_submission.py`
- Data: `data/train.parquet`, `data/test.parquet`
- Current model: LightGBM + XGBoost ensemble with per-horizon training
- Target: `y_target` column
- Weights: `weight` column (extreme skew, use clipping at 99.9 percentile)

## Git Workflow
- Work on branch: `ralph-improvements` (create if not exists)
- Each significant change should be a logical commit with descriptive message
- Commit after each code modification before re-running

## Workflow (Repeat Until Complete)
1. **Ensure on correct branch**: `git checkout ralph-improvements` or `git checkout -b ralph-improvements`
2. **Run forecasting.py**: `marimo run forecasting.py` (runs headlessly, outputs to terminal)
3. **Check the score**: Look for "OVERALL WEIGHTED SCORE" in output - this is the metric to improve
4. **Run plot_submission.py**: `marimo run plot_submission.py` to generate visualization
5. **Analyze**: Review score breakdown by horizon, prediction stats, and plot
6. **Improve**: Modify forecasting.py based on findings
7. **Commit changes**: `git add forecasting.py && git commit -m "score: X.XXXXXX - description of improvement"`
8. **Repeat**: Go back to step 2

## Improvement Ideas to Try
- Feature engineering: more lags, different window sizes, interactions
- Model tuning: learning rate, tree depth, regularization
- Ensemble: try CatBoost, different blend weights
- Target encoding improvements
- Handle cold-start better
- Weight transformation (log1p)
- Different validation split strategy

## Success Criteria
- Internal weighted score > 0.25
- Plot shows reasonable forecast continuation (no extreme spikes or flat lines)
- Predictions have similar distribution to training target

## Commit Convention
Always include the score in commit messages:
```
git commit -m "score: 0.XXXXXX - brief description of what was changed"
```
Example: `git commit -m "score: 0.184523 - added rolling std features for top 20 variables"`

## Completion
When BOTH criteria are met AND you have actually run the code and verified the score, output exactly:
<promise>COMPLETE</promise>

**IMPORTANT**: 
- Do NOT output COMPLETE until you have actually executed forecasting.py and seen a score > 0.25
- Do NOT write the completion promise in any file - only output it in your final response
- Running `marimo run forecasting.py` will execute the notebook headlessly and print results to terminal

## Current Status
Check .ralph/ralph-context.md for any mid-loop hints.
