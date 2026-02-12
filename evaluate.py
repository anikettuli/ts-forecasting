import polars as pl
import numpy as np
import sys
import os

def _clip01(x: float) -> float:
    return float(np.minimum(np.maximum(x, 0.0), 1.0))

def weighted_rmse_score(y_target, y_pred, w) -> float:
    """Calculates the Skill Score: sqrt(1 - sum(w*(y-y_hat)^2) / sum(w*y^2))."""
    denom = np.sum(w * y_target ** 2)
    
    if denom == 0:
        return 0.0
        
    ratio = np.sum(w * (y_target - y_pred) ** 2) / denom
    clipped = _clip01(ratio)
    val = 1.0 - clipped
    return float(np.sqrt(val))

def main(submission_path, data_path="data/test.parquet"):
    if not os.path.exists(submission_path):
        print(f"Error: Submission file '{submission_path}' not found.")
        return

    if not os.path.exists(data_path):
        print(f"Error: Data file '{data_path}' not found. Please download the dataset.")
        return

    print(f"Loading submission: {submission_path}")
    
    sub_df = None
    # Try 1: Standard comma-separated with header (common)
    try:
        sub_df = pl.read_csv(submission_path, separator=",", has_header=True)
        if sub_df.width >= 2:
            # Keep only the first two columns (id, prediction)
            sub_df = sub_df.select(sub_df.columns[:2])
            sub_df.columns = ["id", "prediction"]
        else:
            sub_df = None
    except Exception:
        sub_df = None

    # Try 2: Quirky semicolon format (Header comma -> Data semicolon)
    if sub_df is None:
        try:
            sub_df = pl.read_csv(
                submission_path,
                separator=";",
                has_header=False,
                skip_rows=1,
                new_columns=["id", "prediction"]
            )
        except Exception as e:
            print(f"Error reading submission file: {e}")
            return

    try:
        # Clean up whitespace and cast types
        sub_df = sub_df.with_columns(
            pl.col("id").cast(pl.Utf8).str.strip_chars(),
            pl.col("prediction").cast(pl.Utf8).str.strip_chars().cast(pl.Float64, strict=False)
        )
             
    except Exception as e:
        print(f"Error cleaning submission data: {e}")
        return

    print(f"Submission shape: {sub_df.shape}")

    print(f"Loading ground truth: {data_path}")
    # We need specific columns. If feature_ch (target) is missing, we can't evaluate.
    required_cols = ["id", "feature_ch", "feature_cg", "horizon"]
    try:
        gt_df = pl.read_parquet(data_path).select(required_cols)
    except Exception as e:
        print(f"Error reading ground truth file: {e}")
        print(f"Ensure '{data_path}' contains columns: {required_cols}")
        return

    # Join on id
    eval_df = gt_df.join(sub_df, on="id", how="inner")

    if eval_df.height == 0:
        print("Error: No matching IDs found between submission and ground truth.")
        print("Submission DataFrame Head:")
        print(sub_df.head())
        print("Ground Truth IDs Head:")
        print(gt_df.head())
        return

    print(f"Evaluating {eval_df.height:,} matching rows...")
    print("-" * 60)

    # Calculate overall score
    y_true = eval_df["feature_ch"].to_numpy()
    y_pred = eval_df["prediction"].to_numpy()
    weights = eval_df["feature_cg"].fill_null(1.0).to_numpy()

    overall_score = weighted_rmse_score(y_true, y_pred, weights)
    
    # Header
    print(f"{'METRIC':<20} | {'SCORE':<15} | {'ROWS':<10}")
    print("-" * 60)
    print(f"{'Overall Skill Score':<20} | {overall_score:.6f}        | {eval_df.height:,}")
    print("-" * 60)
    print("Scores by Horizon:")

    # Horizon-specific scores
    horizons = sorted(eval_df["horizon"].unique().to_list())
    for h in horizons:
        h_df = eval_df.filter(pl.col("horizon") == h)
        h_score = weighted_rmse_score(
            h_df["feature_ch"].to_numpy(),
            h_df["prediction"].to_numpy(),
            h_df["feature_cg"].fill_null(1.0).to_numpy()
        )
        print(f"  Horizon {h:<2}         | {h_score:.6f}        | {h_df.height:,}")
    print("-" * 60)

if __name__ == "__main__":
    sub_file = "submission_final_polars.csv" if len(sys.argv) < 2 else sys.argv[1]
    main(sub_file)