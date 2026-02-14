"""Ultra-light analysis - minimal memory, one stat at a time."""

import polars as pl
import sys

sys.stdout.reconfigure(line_buffering=True)  # Force flush after each print

print("=== STEP 1: Schema ===", flush=True)
tr_schema = pl.scan_parquet("data/train.parquet").collect_schema()
te_schema = pl.scan_parquet("data/test.parquet").collect_schema()
print(f"Train cols: {tr_schema.names()}", flush=True)
print(f"Train types: {tr_schema}", flush=True)
print(f"Test cols: {te_schema.names()}", flush=True)
print(f"Test types: {te_schema}", flush=True)

print("\n=== STEP 2: Index ranges ===", flush=True)
tr = pl.scan_parquet("data/train.parquet")
tr_min = tr.select(pl.col("ts_index").min()).collect().item()
tr_max = tr.select(pl.col("ts_index").max()).collect().item()
tr_rows = tr.select(pl.len()).collect().item()
print(f"Train: {tr_rows:,} rows, ts_index=[{tr_min}, {tr_max}]", flush=True)

te = pl.scan_parquet("data/test.parquet")
te_min = te.select(pl.col("ts_index").min()).collect().item()
te_max = te.select(pl.col("ts_index").max()).collect().item()
te_rows = te.select(pl.len()).collect().item()
print(f"Test: {te_rows:,} rows, ts_index=[{te_min}, {te_max}]", flush=True)
print(f"Gap: {te_min - tr_max}", flush=True)

print("\n=== STEP 3: Horizons ===", flush=True)
horizons = te.select(pl.col("horizon").unique().sort()).collect().to_series().to_list()
print(f"Unique horizons: {horizons}", flush=True)

for h in horizons:
    cnt = te.filter(pl.col("horizon") == h).select(pl.len()).collect().item()
    anchor_max = (
        te.filter(pl.col("horizon") == h)
        .select((pl.col("ts_index") - h).max())
        .collect()
        .item()
    )
    print(
        f"  h={h}: {cnt:,} rows, max_anchor={anchor_max}, in_train={anchor_max <= tr_max}",
        flush=True,
    )

print("\n=== STEP 4: Target stats ===", flush=True)
stats = tr.select(
    [
        pl.col("y_target").mean().alias("mean"),
        pl.col("y_target").std().alias("std"),
        pl.col("y_target").min().alias("min"),
        pl.col("y_target").max().alias("max"),
        pl.col("y_target").median().alias("median"),
        (pl.col("y_target") == 0).mean().alias("frac_zero"),
        (pl.col("y_target") < 0).mean().alias("frac_neg"),
    ]
).collect()
print(stats, flush=True)

print("\n=== STEP 5: Weight stats ===", flush=True)
wstats = tr.select(
    [
        pl.col("weight").mean().alias("mean"),
        pl.col("weight").std().alias("std"),
        pl.col("weight").min().alias("min"),
        pl.col("weight").max().alias("max"),
        pl.col("weight").n_unique().alias("n_unique"),
    ]
).collect()
print(wstats, flush=True)

print("\n=== STEP 6: Group counts ===", flush=True)
for c in ["code", "sub_code", "sub_category"]:
    n = tr.select(pl.col(c).n_unique()).collect().item()
    print(f"  {c}: {n} unique", flush=True)

print("\n=== STEP 7: Submissions ===", flush=True)
for f in ["submission_optimized.csv", "submission_optimized99.csv"]:
    try:
        with open(f, "r") as fh:
            line1 = fh.readline()
        sep = ";" if ";" in line1 else ","
        df = pl.read_csv(f, separator=sep, n_rows=50000)
        c = df.columns
        pred = df[c[-1]].cast(pl.Float64, strict=False).fill_null(0.0)
        print(
            f"{f}: mean={pred.mean():.6f} std={pred.std():.6f} min={pred.min():.6f} max={pred.max():.6f}",
            flush=True,
        )
    except Exception as ex:
        print(f"  {f}: {ex}", flush=True)

print("\n✅ DONE", flush=True)
