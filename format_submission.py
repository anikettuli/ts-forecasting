import pandas as pd
import sys
import polars as pl
import os

def convert_submission_format(input_path="submission_optimized.csv", output_path="submission_semicolon.csv"):
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        return

    print("Reading CSV...")
    # Use Polars for speed
    df = pl.read_csv(input_path)
    
    print(f"Writing {len(df)} rows to '{output_path}'...")
    
    # We need:
    # Header: "id,prediction"
    # Rows:   "ID ; VALUE"
    
    with open(output_path, 'w') as f:
        f.write("id,prediction\n")
    
    # Create the formatted string column efficiently
    # Using Polars string concatenation is very fast
    # Format: id + " ; " + prediction
    
    output_df = df.select(
        (pl.col("id") + pl.lit(" ; ") + pl.col("prediction").cast(pl.Utf8)).alias("line")
    )
    
    # Write directly without header (since we wrote it manually)
    # Using to_csv with no header, no index
    # But Polars to_csv might quote strings by default
    
    # Let's write manually in chunks if needed, or use pandas for the final write if safer
    # Actually, writing a single column CSV with no header and no quotes is easiest
    
    output_df.write_csv(
        output_path,
        include_header=False,
        separator="\n", # Hack: separate "columns" by newline? No, valid separator is needed.
        quote_style="never"
    )
    
    # Wait, write_csv writes columns separated by separator.
    # Since we have 1 column "line", the separator doesn't matter much unless there are special chars.
    # We want to append to the file we just created?
    # verify if write_csv supports append. Polars doesn't support append mode easily.
    
    # Alternative: Use Python's file write with a generator
    with open(output_path, 'w') as f:
        f.write("id,prediction\n")
        # Generator for lines
        lines = (f"{row[0]} ; {row[1]}\n" for row in df.iter_rows())
        f.writelines(lines)

    print(f"Successfully converted '{input_path}' to '{output_path}'")
    
    # Verification
    with open(output_path, 'r') as f:
        print("First 3 lines:")
        for _ in range(3):
            print(f.readline().strip())

if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "submission_optimized.csv"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "submission_semicolon.csv"
    convert_submission_format(input_file, output_file)
