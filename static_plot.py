import sys
import matplotlib.pyplot as plt
import polars as pl
import os

sub_path = sys.argv[1]
sub_df = pl.read_csv(sub_path)
sub_df.columns = ["id", "prediction"]
sub_df = sub_df.with_columns(pl.col("id").cast(pl.Utf8).str.strip_chars())

test_meta = pl.read_parquet("data/test.parquet").select(["id", "ts_index", "code"])
joined = sub_df.join(test_meta, on="id", how="inner")

p_data = joined.group_by("ts_index").agg(pl.col("prediction").sum()).sort("ts_index")

train_df = pl.read_parquet("data/train.parquet")
h_data = train_df.group_by("ts_index").agg(pl.col("y_target").sum()).sort("ts_index")

plt.figure(figsize=(12, 6))
plt.plot(h_data["ts_index"], h_data["y_target"], label="Historical", color="blue")
plt.plot(
    p_data["ts_index"],
    p_data["prediction"],
    label="Forecast",
    color="red",
    linestyle="--",
)
plt.title("Global Aggregate Forecast")
plt.legend()
plt.savefig("forecast.png")
print("Saved forecast.png")
