import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    GROUP_COLS = ["code", "sub_code", "sub_category"]
    return GROUP_COLS, mo, pl


@app.cell
def _(GROUP_COLS, mo, pl):
    mo.md("## 1. Raw Data Structure")

    train = pl.scan_parquet("data/train.parquet")
    test = pl.scan_parquet("data/test.parquet")

    print("Train Schema:", train.collect_schema())
    print("Test Schema:", test.collect_schema())

    tr_stats = train.select(
        [
            pl.len().alias("n_rows"),
            pl.col("ts_index").min().alias("ts_min"),
            pl.col("ts_index").max().alias("ts_max"),
            pl.col("ts_index").n_unique().alias("ts_unique"),
            pl.col("y_target").mean().alias("y_mean"),
            pl.col("y_target").std().alias("y_std"),
            pl.col("y_target").min().alias("y_min"),
            pl.col("y_target").max().alias("y_max"),
            pl.col("y_target").median().alias("y_median"),
            (pl.col("y_target") == 0).mean().alias("pct_zero"),
            (pl.col("y_target") < 0).mean().alias("pct_negative"),
        ]
    ).collect()

    te_stats = test.select(
        [
            pl.len().alias("n_rows"),
            pl.col("ts_index").min().alias("ts_min"),
            pl.col("ts_index").max().alias("ts_max"),
            pl.col("ts_index").n_unique().alias("ts_unique"),
            pl.col("horizon").min().alias("h_min"),
            pl.col("horizon").max().alias("h_max"),
        ]
    ).collect()

    print("\n--- TRAIN ---")
    print(tr_stats)
    print("\n--- TEST ---")
    print(te_stats)

    gap = te_stats["ts_min"][0] - tr_stats["ts_max"][0]
    print(f"\nGap between train end and test start: {gap} days")

    tr_max = tr_stats["ts_max"][0]

    # Group structure
    for c in GROUP_COLS:
        n_tr = train.select(pl.col(c).n_unique()).collect().item()
        n_te = test.select(pl.col(c).n_unique()).collect().item()
        print(f"  {c}: train={n_tr}, test={n_te}")

    n_series_tr = train.select(GROUP_COLS).unique().collect().height
    n_series_te = test.select(GROUP_COLS).unique().collect().height
    print(f"Total unique series: train={n_series_tr}, test={n_series_te}")

    return gap, te_stats, test, tr_max, tr_stats, train


@app.cell
def _(mo, pl, test, tr_max):
    mo.md("## 2. Horizon & Anchor Analysis (THE KEY ISSUE)")

    horizon_analysis = (
        test.group_by("horizon")
        .agg(
            [
                pl.col("ts_index").min().alias("ts_min"),
                pl.col("ts_index").max().alias("ts_max"),
                pl.len().alias("n_rows"),
            ]
        )
        .with_columns(
            [
                (pl.col("ts_min") - pl.col("horizon")).alias("anchor_min"),
                (pl.col("ts_max") - pl.col("horizon")).alias("anchor_max"),
            ]
        )
        .with_columns(
            [
                (pl.col("anchor_max") <= tr_max).alias("anchor_in_train"),
            ]
        )
        .sort("horizon")
        .collect()
    )

    print(f"Train max ts_index: {tr_max}")
    print(horizon_analysis)

    n_outside = (
        test.filter((pl.col("ts_index") - pl.col("horizon")) > tr_max)
        .select(pl.len())
        .collect()
        .item()
    )

    n_total = test.select(pl.len()).collect().item()
    print(
        f"\nTest rows with anchor OUTSIDE train: {n_outside:,} / {n_total:,} "
        f"({n_outside / n_total * 100:.1f}%)"
    )
    return horizon_analysis, n_outside, n_total


@app.cell
def _(GROUP_COLS, mo, pl, train):
    mo.md("## 3. Target Distribution & Per-Series Variation")

    quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    q_exprs = [
        pl.col("y_target").quantile(q).alias(f"P{int(q * 100):02d}") for q in quantiles
    ]
    q_df = train.select(q_exprs).collect()
    print("Target Quantiles:")
    print(q_df)

    w_stats = train.select(
        [
            pl.col("weight").mean().alias("w_mean"),
            pl.col("weight").std().alias("w_std"),
            pl.col("weight").min().alias("w_min"),
            pl.col("weight").max().alias("w_max"),
            (pl.col("weight") == 0).mean().alias("w_pct_zero"),
            pl.col("weight").n_unique().alias("w_unique"),
        ]
    ).collect()
    print("\nWeight Stats:")
    print(w_stats)

    series_stats = (
        train.group_by(GROUP_COLS)
        .agg(
            [
                pl.col("y_target").mean().alias("mean"),
                pl.col("y_target").std().alias("std"),
                pl.col("y_target").min().alias("min"),
                pl.col("y_target").max().alias("max"),
                pl.len().alias("n_obs"),
            ]
        )
        .sort("std", descending=True)
        .collect()
    )
    print(f"\nTotal series: {series_stats.height}")
    print("\nTop 10 most volatile series:")
    print(series_stats.head(10))
    print("\nBottom 10 flattest series:")
    print(series_stats.tail(10))
    return q_df, series_stats, w_stats


@app.cell
def _(mo, pl, train, tr_max):
    mo.md("## 4. Last Window of Training (What the model sees at forecast time)")

    last_days = train.filter(pl.col("ts_index") >= tr_max - 50)
    global_agg = (
        last_days.group_by("ts_index")
        .agg(
            [
                pl.col("y_target").sum().alias("y_sum"),
                pl.col("y_target").mean().alias("y_mean"),
                pl.col("y_target").std().alias("y_std"),
            ]
        )
        .sort("ts_index")
        .collect()
    )

    print("Last 30 days - Global sum/mean/std:")
    print(global_agg.tail(30))

    one_series = (
        train.sort("ts_index")
        .filter(pl.col("ts_index") >= tr_max - 100)
        .head(500)
        .collect()
    )
    if one_series.height > 0:
        codes = one_series.select(["code", "sub_code", "sub_category"]).unique()
        print("\nSample series (first in last 100 days):")
        print(codes.head(1))
    return global_agg, one_series


@app.cell
def _(mo, pl):
    mo.md("## 5. Submission File Comparison")

    for fname in ["submission_optimized.csv", "submission_optimized99.csv"]:
        try:
            with open(fname, "r") as fh:
                line = fh.readline()
            sep = ";" if ";" in line else ","
            sub_df = pl.read_csv(fname, separator=sep, n_rows=100000)
            cols = sub_df.columns
            pred = sub_df[cols[-1]].cast(pl.Float64, strict=False).fill_null(0.0)
            print(f"\n{'=' * 50}")
            print(f"{fname}: (first 100K rows)")
            print(f"  Sep: '{sep}', Cols: {cols}")
            print(f"  Mean: {pred.mean():.6f}, Std: {pred.std():.6f}")
            print(f"  Min: {pred.min():.6f}, Max: {pred.max():.6f}")
            print(
                f"  Zero%: {(pred == 0).mean() * 100:.1f}%, Neg%: {(pred < 0).mean() * 100:.1f}%"
            )
        except Exception:
            print(f"  Error loading {fname}")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Diagnosis Summary

    **After reviewing all the data above, here is what we know:**

    1. **The Global Aggregate plot sums ALL series** — so even if each individual series is small,
       summing hundreds of series creates the ±20,000 swings you see in the history.
    2. **The model predicts per-series** — each individual prediction is tiny (e.g., 0.001 to 3.0),
       which is correct for a single series, but when summed, it should still create variation.
    3. **The flat forecast** means either:
       - All series get the same constant prediction (features are identical → all-zero from failed joins), or
       - The model lacks variation in inputs (cyclical features alone can't drive ±20,000 swings).

    **The fix** requires ensuring that each test series receives its OWN last-known lag values
    (not clamped to a single global timestamp), and that the model has enough signal to differentiate
    between volatile and stable series.
    """)
    return


if __name__ == "__main__":
    app.run()
