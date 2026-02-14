import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import matplotlib.pyplot as plt
    import os
    import glob

    return mo, pl, plt, os, glob


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # 📈 Forecast Intelligence Dashboard
        Visualize individual series or the **global aggregate trend**.
        """
    )
    return


@app.cell
def _(glob, mo):
    # Find all CSV files in the current directory
    csv_files = glob.glob("*.csv")
    csv_files = [f for f in csv_files if "validation" not in f.lower()]

    sub_file_select = mo.ui.dropdown(
        options=csv_files,
        value="submission_optimized.csv"
        if "submission_optimized.csv" in csv_files
        else (csv_files[0] if csv_files else None),
        label="Select Submission File:",
    )

    sub_file_select
    return csv_files, sub_file_select


@app.cell
def _(mo, os, pl, sub_file_select):
    mo.stop(sub_file_select.value is None, "No submission file selected.")

    # Paths
    DATA_DIR = "data"
    TRAIN_PATH = os.path.join(DATA_DIR, "train.parquet")
    TEST_PATH = os.path.join(DATA_DIR, "test.parquet")
    SUB_PATH = sub_file_select.value

    # Load Data
    train_lf = pl.scan_parquet(TRAIN_PATH)
    test_meta_lf = pl.scan_parquet(TEST_PATH)

    # Robust CSV Loader (tries to match evaluate.py logic)
    try:
        # Try reading with stripping whitespace and explicit casting
        raw_sub = pl.read_csv(SUB_PATH, truncate_ragged_lines=True)
        # Standardize columns to id and prediction
        if raw_sub.width >= 2:
            cols = raw_sub.columns
            sub_df = raw_sub.select(
                [
                    pl.col(cols[0]).cast(pl.Utf8).str.strip_chars().alias("id"),
                    pl.col(cols[1]).cast(pl.Float64, strict=False).alias("prediction"),
                ]
            )
        else:
            sub_df = pl.DataFrame({"id": [], "prediction": []})
    except Exception:
        sub_df = pl.DataFrame({"id": [], "prediction": []})

    sub_df = sub_df.fill_null(0.0)

    unique_codes = sorted(train_lf.select("code").unique().collect()["code"].to_list())
    return SUB_PATH, TEST_PATH, TRAIN_PATH, sub_df, train_lf, test_meta_lf, unique_codes


@app.cell
def _(mo, unique_codes):
    view_mode = mo.ui.radio(
        options=["Global Aggregate", "Single Code View"],
        value="Global Aggregate",
        label="View Mode:",
    )

    code_select = mo.ui.dropdown(
        options=unique_codes,
        value=unique_codes[0] if unique_codes else None,
        label="Select Code:",
        searchable=True,
    )
    return code_select, view_mode


@app.cell
def _(code_select, mo, view_mode):
    if view_mode.value == "Single Code View":
        display_ui = mo.hstack([view_mode, code_select])
    else:
        display_ui = view_mode
    display_ui
    return (display_ui,)


@app.cell
def _(code_select, mo, pl, sub_df, plt, test_meta_lf, train_lf, view_mode):
    # Prepare Test Meta with cleaned IDs for joining
    p_meta = test_meta_lf.select(
        [
            pl.col("id").cast(pl.Utf8).str.strip_chars(),
            pl.col("ts_index"),
            pl.col("code"),
        ]
    ).collect()

    if view_mode.value == "Global Aggregate":
        title = "Global Aggregate: Total History vs Total Forecast"
        h_data = (
            train_lf.group_by("ts_index")
            .agg(pl.col("y_target").sum())
            .collect()
            .sort("ts_index")
        )

        # Match Predictions
        joined = sub_df.join(p_meta, on="id", how="inner")
        if joined.height == 0:
            p_data = pl.DataFrame({"ts_index": [], "prediction": []})
            error_msg = mo.md(
                "⚠️ **Warning: No matching IDs found in submission!** Check if your IDs match the test set."
            )
        else:
            p_data = (
                joined.group_by("ts_index")
                .agg(pl.col("prediction").sum())
                .sort("ts_index")
            )
            error_msg = None
    else:
        mo.stop(code_select.value is None, "Please select a code.")
        selected_code = code_select.value
        title = f"Series View: Code {selected_code}"

        h_data = (
            train_lf.filter(pl.col("code") == selected_code)
            .select(["ts_index", "y_target"])
            .collect()
            .sort("ts_index")
        )

        code_meta = p_meta.filter(pl.col("code") == selected_code)
        joined = sub_df.join(code_meta, on="id", how="inner")
        p_data = joined.sort("ts_index")
        error_msg = (
            mo.md("⚠️ **Warning: No predictions found for this code.**")
            if joined.height == 0
            else None
        )

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        h_data["ts_index"],
        h_data["y_target"],
        label="Historical",
        color="#636EFA",
        linewidth=2,
    )

    if p_data.height > 0:
        ax.plot(
            p_data["ts_index"],
            p_data["prediction"],
            label="Forecast",
            color="#EF553B",
            linestyle="--",
            marker="o",
            markersize=3,
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    return ax, error_msg, fig, h_data, p_data


@app.cell
def _(error_msg, fig, mo):
    mo.vstack([error_msg if error_msg else mo.md(""), mo.as_html(fig)])
    return


if __name__ == "__main__":
    app.run()
