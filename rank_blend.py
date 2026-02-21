import polars as pl
import numpy as np
from scipy.stats import rankdata

print("Loading submissions for Rank Blending...")
best_sub = pl.read_csv("submission_optimized.csv")
new_sub = pl.read_csv("submission.csv")

if best_sub.shape[0] != new_sub.shape[0]:
    print("Shapes don't match, attempting ID alignment...")
    best_sub = best_sub.sort("id")
    new_sub = new_sub.sort("id")

# Weights: prioritize the optimized 0.25+ submission but pull signal from our cross-sectional code
w_best = 0.65
w_new = 0.35

print(f"Blending with {w_best} / {w_new} ratio using Rank Transformation...")

p1 = best_sub["prediction"].to_numpy()
p2 = new_sub["prediction"].to_numpy()

# Rank transform both to normalize distributions and prevent scale mismatch
rank1 = rankdata(p1) / len(p1)
rank2 = rankdata(p2) / len(p2)

# Blend ranks
blended_ranks = (w_best * rank1) + (w_new * rank2)

# Inverse transform back to the distribution of the 'best' submission (to preserve scale)
sorted_p1 = np.sort(p1)
indices = (blended_ranks * (len(p1) - 1)).astype(int)
final_preds = sorted_p1[indices]

# Save final ensemble
final_sub = new_sub.with_columns(pl.Series("prediction", final_preds))
final_sub.write_csv("submission_super_ensemble.csv")

print("Saved submission_super_ensemble.csv")
print(f"Stats - Best Sub: Mean={np.mean(p1):.4f}, Std={np.std(p1):.4f}")
print(f"Stats - New Sub: Mean={np.mean(p2):.4f}, Std={np.std(p2):.4f}")
print(
    f"Stats - Ensemble: Mean={np.mean(final_preds):.4f}, Std={np.std(final_preds):.4f}"
)
