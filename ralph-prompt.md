# Kaggle TS-Forecasting Competition - Iterative Improvement Task

## Your Task
Improve the time series forecasting model to achieve an internal weighted score above 0.25.

## Instructions
1. First, checkout or create branch: `git checkout -b ralph-improvements`
2. Run the forecasting notebook: `marimo run forecasting.py`
3. Find the "OVERALL WEIGHTED SCORE" in the output
4. Analyze what could be improved
5. Make code changes to forecasting.py
6. Commit with score: `git commit -m "score: X.XXXXXX - description"`
7. Repeat until score > 0.25

## Key Files
- `forecasting.py` - main model (marimo notebook)
- `plot_submission.py` - visualization
- `data/train.parquet`, `data/test.parquet` - data
- Target column: `y_target`, Weight column: `weight`

## Success
Output `<promise>COMPLETE</promise>` ONLY after:
- Running `marimo run forecasting.py`
- Seeing "OVERALL WEIGHTED SCORE" > 0.25
- Verifying the plot makes sense with `marimo run plot_submission.py`
