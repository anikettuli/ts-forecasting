#!/usr/bin/env python3
"""Local evaluator for ts-forecasting competition.

Usage:
    python evaluate.py submission.csv              # Checks format + scores against train
    python evaluate.py submission.csv --test-only  # Only checks format/IDs

Note: Kaggle hides y_target and weight in test.parquet. 
This script validates format and can score against train.parquet for debugging.
"""

import argparse
import os
import sys

import numpy as np
import polars as pl


def _clip01(x: float) -> float:
    """Clip value to [0, 1] range."""
    return float(np.minimum(np.maximum(x, 0.0), 1.0))


def weighted_rmse_score(y_target, y_pred, w) -> float:
    """Official Kaggle Weighted RMSE Skill Score.
    
    Formula: sqrt(1 - clip(sum(w*(y-y_hat)^2) / sum(w*y^2)))
    Score of 1.0 = perfect prediction, 0.0 = worst
    """
    denom = np.sum(w * y_target ** 2)
    ratio = np.sum(w * (y_target - y_pred) ** 2) / denom
    clipped = _clip01(ratio)
    val = 1.0 - clipped
    return float(np.sqrt(val))


def load_submission(path: str) -> pl.DataFrame:
    """Load submission CSV, handling common format variations."""
    try:
        df = pl.read_csv(path, separator=",", has_header=True)
        if df.width >= 2:
            df = df.select(df.columns[:2])
            df.columns = ["id", "prediction"]
            return df
    except Exception:
        pass
    
    try:
        df = pl.read_csv(path, separator=";", has_header=False, skip_rows=1, new_columns=["id", "prediction"])
        return df
    except Exception as e:
        print(f"Error: Cannot parse submission file: {e}")
        sys.exit(1)


def check_submission_format(sub_df: pl.DataFrame, test_ids: set) -> list:
    """Validate submission format and completeness."""
    issues = []
    
    if list(sub_df.columns) != ["id", "prediction"]:
        issues.append(f"Wrong columns: {sub_df.columns} (expected ['id', 'prediction'])")
    
    sub_ids = set(sub_df["id"].to_list())
    missing = test_ids - sub_ids
    extra = sub_ids - test_ids
    
    if missing:
        issues.append(f"Missing {len(missing)} test IDs")
    if extra:
        issues.append(f"Extra {len(extra)} IDs not in test set")
    
    null_count = sub_df["prediction"].null_count()
    if null_count > 0:
        issues.append(f"{null_count} null predictions")
    
    return issues


def main():
    parser = argparse.ArgumentParser(description="Evaluate submission for ts-forecasting")
    parser.add_argument("submission", help="Path to submission CSV file")
    parser.add_argument("--test-only", action="store_true", help="Only check format, don't score")
    parser.add_argument("--train", default="data/train.parquet", help="Path to train parquet")
    parser.add_argument("--test", default="data/test.parquet", help="Path to test parquet")
    args = parser.parse_args()
    
    # Validate files exist
    if not os.path.exists(args.submission):
        print(f"Error: Submission file not found: {args.submission}")
        sys.exit(1)
    if not os.path.exists(args.test):
        print(f"Error: Test file not found: {args.test}")
        sys.exit(1)
    
    # Load submission
    print(f"Loading submission: {args.submission}")
    sub_df = load_submission(args.submission)
    sub_df = sub_df.with_columns(
        pl.col("id").cast(pl.Utf8).str.strip_chars(),
        pl.col("prediction").cast(pl.Float64, strict=False)
    )
    print(f"  Shape: {sub_df.shape}")
    
    # Load test IDs for validation
    print(f"Loading test data: {args.test}")
    test_df = pl.read_parquet(args.test).select(["id", "horizon"])
    test_ids = set(test_df["id"].to_list())
    print(f"  Test IDs: {len(test_ids):,}")
    
    # Check format
    issues = check_submission_format(sub_df, test_ids)
    if issues:
        print("\nFormat Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nFormat: OK")
    
    if args.test_only:
        print("\nFormat check complete.")
        return
    
    # Local scoring against train data
    print(f"\nLoading train data: {args.train}")
    if not os.path.exists(args.train):
        print("Error: Train file not found")
        sys.exit(1)
    
    train_df = pl.read_parquet(args.train).select(["id", "y_target", "weight", "horizon"])
    print(f"  Train rows: {train_df.height:,}")
    
    eval_df = train_df.join(sub_df, on="id", how="inner")
    
    if eval_df.height == 0:
        print("\nNote: Submission IDs match test set (not train).")
        print("Submit to Kaggle for official evaluation.")
        return
    
    # Calculate scores
    y_true = eval_df["y_target"].to_numpy()
    y_pred = eval_df["prediction"].fill_null(0.0).to_numpy()
    weights = eval_df["weight"].fill_null(1.0).to_numpy()
    
    overall_score = weighted_rmse_score(y_true, y_pred, weights)
    
    print("-" * 60)
    print(f"{'METRIC':<20} | {'SCORE':<15} | {'ROWS':<10}")
    print("-" * 60)
    print(f"{'Local Skill Score':<20} | {overall_score:.6f}        | {eval_df.height:,}")
    print("-" * 60)
    print("Scores by Horizon:")
    
    for h in sorted(eval_df["horizon"].unique().to_list()):
        h_df = eval_df.filter(pl.col("horizon") == h)
        h_score = weighted_rmse_score(
            h_df["y_target"].to_numpy(),
            h_df["prediction"].fill_null(0.0).to_numpy(),
            h_df["weight"].fill_null(1.0).to_numpy()
        )
        print(f"  Horizon {h:<2}         | {h_score:.6f}        | {h_df.height:,}")
    
    print("-" * 60)


if __name__ == "__main__":
    main()
