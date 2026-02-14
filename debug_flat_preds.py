import polars as pl
import numpy as np

# Load data
train = pl.scan_parquet("data/train.parquet")
test = pl.scan_parquet("data/test.parquet")

max_ts = train.select(pl.col("ts_index").max()).collect().item()
test_min_ts = test.select(pl.col("ts_index").min()).collect().item()

print(f"Train Max Index: {max_ts}")
print(f"Test Min Index: {test_min_ts}")

# Check first test row horizon
test_row = test.head(1).collect()
h = test_row["horizon"][0]
ts = test_row["ts_index"][0]

join_ts_old = ts - h + 1
join_ts_new = ts - h

print(f"First Test Row: ts_index={ts}, horizon={h}")
print(f"Current logic join_ts: {join_ts_old} (Is in train? {join_ts_old <= max_ts})")
print(f"Proposed logic join_ts: {join_ts_new} (Is in train? {join_ts_new <= max_ts})")

# Check full submission variation
sub = pl.read_csv("submission_optimized.csv")
print("\nSubmission Statistics:")
print(
    sub.select(
        [
            pl.col("prediction").mean().alias("mean"),
            pl.col("prediction").std().alias("std"),
            pl.col("prediction").n_unique().alias("unique_values"),
        ]
    )
)
