"""
Iteration E: Enhanced Target Encoding with Smoothing (Polars)
"""

import polars as pl
import numpy as np
import warnings

warnings.filterwarnings("ignore")

def create_smoothed_target_encoding(df, col, target="feature_ch", weight="feature_cg", smoothing=10):
    df = df.sort([col, "ts_index"])
    global_mean = df[target].mean()
    
    # Smoothed Expanding Mean
    return df.with_columns(
        (
            (pl.col(target).shift(1).cum_sum().over(col).fill_null(0) + smoothing * global_mean) 
            / 
            (pl.col(target).shift(1).cum_count().over(col).fill_null(0) + smoothing)
        ).alias(f"{col}_enc_smooth")
    )

if __name__ == "__main__":
    print("Iteration E: Smoothed Encoding (Polars)")
    df = pl.read_parquet("data/test.parquet")
    
    for col in ["code", "sub_code", "sub_category"]:
        print(f"Encoding {col}...")
        df = create_smoothed_target_encoding(df, col)
        
    print("Done.")
