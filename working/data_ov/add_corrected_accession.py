"""Map untruncated accession numbers onto an existing overview CSV.

This is the fast counterpart to ``check_unavailable_images.py``: it skips the
slow nifti-tree walk entirely and only joins the corrected accession numbers
(from a source CSV) onto an overview CSV that already has the ``downloaded``
column. Use it when you just need to (re)add ``accession_number_corrected``
without recomputing the download status.

The join is on ``info_key`` + ``sip`` and the corrected values land in a new
``accession_number_corrected`` column; the corrupted original is left as-is.
"""

import argparse
from pathlib import Path

import pandas as pd

from check_unavailable_images import (
    ACCESSION_KEYS,
    ACCESSION_OUT_COL,
    merge_corrected_accession,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("overview_csv", type=Path,
                    help="existing overview CSV (e.g. kira-0515-seg-downloaded.csv)")
    ap.add_argument("accession_csv", type=Path,
                    help="CSV with untruncated accession numbers")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output CSV (default: overwrite the overview CSV in place)")
    args = ap.parse_args()

    for p in (args.overview_csv, args.accession_csv):
        if not p.exists():
            raise SystemExit(f"file not found: {p}")

    df = pd.read_csv(args.overview_csv)
    missing_keys = [c for c in ACCESSION_KEYS if c not in df.columns]
    if missing_keys:
        raise SystemExit(
            f"join key(s) {', '.join(missing_keys)} not in {args.overview_csv.name}"
        )

    n_corrected = merge_corrected_accession(df, args.accession_csv)

    out = args.out or args.overview_csv
    df.to_csv(out, index=False)
    print(f"Corrected accession numbers ('{ACCESSION_OUT_COL}'): "
          f"{n_corrected} / {len(df)} rows matched in {args.accession_csv.name}")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
