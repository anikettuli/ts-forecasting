import polars as pl
import numpy as np
import os

files = ["submission_optimized.csv", "submission_optimized99.csv"]

for f in files:
    if not os.path.exists(f):
        print(f"{f}: Not found")
        continue
    try:
        # Read only the prediction column
        df = pl.read_csv(f).select(pl.all().exclude("id"))
        pred_col = df.columns[0]
        preds = df[pred_col].cast(pl.Float64, strict=False).fill_null(0.0).to_numpy()

        std = np.std(preds)
        mean = np.mean(preds)
        min_v = np.min(preds)
        max_v = np.max(preds)
        unique_vals = len(np.unique(preds))

        print(f"File: {f}")
        print(f"  Mean: {mean:.4f}")
        print(f"  Std:  {std:.4f}")
        print(f"  Min:  {min_v:.4f}")
        print(f"  Max:  {max_v:.4f}")
        print(f"  Unique Values: {unique_vals}")
        print("-" * 30)
    except Exception as e:
        print(f"Error reading {f}: {e}")
