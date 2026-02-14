import polars as pl
import numpy as np
import os


def check_file(f):
    if not os.path.exists(f):
        return f"{f}: Not found"
    try:
        # Detect separator
        with open(f, "r") as fh:
            first_line = fh.readline()
            sep = ";" if ";" in first_line else ","

        df = pl.read_csv(f, separator=sep)
        pred_col = [
            c for c in df.columns if "pred" in c.lower() or c == df.columns[-1]
        ][0]
        preds = df[pred_col].cast(pl.Float64, strict=False).fill_null(0.0).to_numpy()

        std = np.std(preds)
        mean = np.mean(preds)
        unique_vals = len(np.unique(preds))

        return f"File: {f} | Mean: {mean:.4f} | Std: {std:.4f} | Unique: {unique_vals}"
    except Exception as e:
        return f"Error {f}: {e}"


files = [
    "submission_optimized.csv",
    "submission_optimized99.csv",
    "validation_results.csv",
]
for f in files:
    print(check_file(f))
