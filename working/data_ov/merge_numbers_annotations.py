import argparse
from pathlib import Path

import numbers_parser
import pandas as pd


def load_numbers_table(path: Path) -> pd.DataFrame:
    doc = numbers_parser.Document(str(path))
    table = doc.sheets[0].tables[0]
    data = table.rows(values_only=True)
    header, rows = data[0], data[1:]
    df = pd.DataFrame(rows, columns=header)
    return df.loc[:, ~df.columns.duplicated()]


def main():
    parser = argparse.ArgumentParser(
        description="Merge If_segmented/note annotations from an edited .numbers file "
        "back into the corresponding .csv, keyed on info_key. Values from the .numbers "
        "file win when both are present and differ."
    )
    parser.add_argument("-n", "--numbers", required=True, help="Edited .numbers file")
    parser.add_argument("-c", "--csv", required=True, help="Target .csv to merge into")
    parser.add_argument("-o", "--output", default=None, help="Output path (default: <csv>.merged.csv)")
    parser.add_argument(
        "--columns", nargs="+", default=["If_segmented", "note"],
        help="Columns to pull from the .numbers file (default: If_segmented note)",
    )
    args = parser.parse_args()

    num_df = load_numbers_table(Path(args.numbers))
    csv_df = pd.read_csv(args.csv)

    merged = csv_df.merge(
        num_df[["info_key", *args.columns]], on="info_key", how="left", suffixes=("_old", "")
    )
    for col in args.columns:
        old_col = f"{col}_old"
        if old_col in merged.columns:
            conflicts = merged[
                merged[old_col].notna() & merged[col].notna() & (merged[old_col] != merged[col])
            ]
            if len(conflicts):
                print(f"{col}: {len(conflicts)} conflict(s), .numbers value wins:")
                print(conflicts[["info_key", old_col, col]].to_string(index=False))
            merged[col] = merged[col].where(merged[col].notna(), merged[old_col])
            merged = merged.drop(columns=[old_col])

    output_path = Path(args.output) if args.output else Path(args.csv).with_suffix(".merged.csv")
    merged.to_csv(output_path, index=False)
    print(f"Wrote merged CSV ({len(merged)} rows) to {output_path}")


if __name__ == "__main__":
    main()
