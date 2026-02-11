import polars as pl
import numpy as np
import sys
import os

def _clip01(x: float) -> float:
    return float(np.minimum(np.maximum(x, 0.0), 1.0))

def weighted_rmse_score(y_target, y_pred, w) -> float:
    """Calculates the Skill Score: 1 - sqrt(sum(w*(y-y_hat)^2) / sum(w*y^2))."""
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
        print(f"Error: Data file '{data_path}' not found.")
        return

    print(f"Loading submission: {submission_path}")
    
    # Try reading with comma, if fail try semicolon
    try:
        sub_df = pl.read_csv(submission_path)
        if "id" not in sub_df.columns or "prediction" not in sub_df.columns:
            raise ValueError("Columns missing")
    except:
        try:
            # Try semicolon separator
            sub_df = pl.read_csv(submission_path, separator=";")
            # Remove whitespace from columns if present
            sub_df = sub_df.with_columns(pl.col("id").str.strip_chars(), pl.col("prediction"))
        except Exception as e:
            print(f"Error reading submission file: {e}")
            return

    print(f"Loading ground truth: {data_path}")
    # We only need meta, target and weights
    gt_df = pl.read_parquet(data_path).select(["id", "feature_ch", "feature_cg", "horizon"])

    # Join on id
    eval_df = gt_df.join(sub_df, on="id", how="inner")

    if eval_df.height == 0:
        print("Error: No matching IDs found between submission and ground truth.")
        return

    print(f"Evaluating {eval_df.height:,} matching rows...")

    # Calculate overall score
    y_true = eval_df["feature_ch"].to_numpy()
    y_pred = eval_df["prediction"].to_numpy()
    weights = eval_df["feature_cg"].fill_null(1.0).to_numpy()

    overall_score = weighted_rmse_score(y_true, y_pred, weights)
    print(f"\n{'='*30}")
    print(f"OVERALL SKILL SCORE: {overall_score:.6f}")
    print(f"{'='*30}\n")

    # Horizon-specific scores
    print("Scores by Horizon:")
    horizons = sorted(eval_df["horizon"].unique().to_list())
    for h in horizons:
        h_df = eval_df.filter(pl.col("horizon") == h)
        h_score = weighted_rmse_score(
            h_df["feature_ch"].to_numpy(),
            h_df["prediction"].to_numpy(),
            h_df["feature_cg"].fill_null(1.0).to_numpy()
        )
        print(f"  Horizon {h:2}: {h_score:.6f} ({h_df.height:,} rows)")

if __name__ == "__main__":
    sub_file = "submission_final_polars.csv" if len(sys.argv) < 2 else sys.argv[1]
    main(sub_file)