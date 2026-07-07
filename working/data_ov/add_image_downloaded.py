"""Add an ``image_downloaded`` column to the original kira CSV.

Rule (per row):
  - image_downloaded = "No"  when the row HAS a subject_code but that subject
    has NO imaging (images.nii.gz) anywhere on disk.
  - image_downloaded = "Yes" otherwise (subject has imaging, or no subject_code).

Run on the machine that holds the data (default root is the remote path).
"""

import argparse
from pathlib import Path

import pandas as pd

from distribution_imaging import DEFAULT_DATA_ROOT, subjects_with_images

REPO_ROOT = Path("/output")
DEFAULT_CSV = REPO_ROOT / "kira-0515-seg.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("csv", nargs="?", type=Path, default=DEFAULT_CSV,
                    help=f"source CSV with subject_code (default: {DEFAULT_CSV})")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                    help="root of the BONE_AI_* data tree to scan for images")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output CSV (default: <csv stem>-image_downloaded.csv)")
    args = ap.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")
    if not args.data_root.exists():
        raise SystemExit(
            f"Data root not found: {args.data_root}\n"
            "Run on the machine that has the data, or pass --data-root."
        )

    df = pd.read_csv(args.csv)
    if "subject_code" not in df.columns:
        raise SystemExit(f"'subject_code' column not in {args.csv.name}")

    img_subjects = subjects_with_images(args.data_root)
    # fillna("") first: read_csv may use the pandas 'str' dtype, where NaN stays
    # a float and astype(str) would NOT turn it into "nan".
    subj = df["subject_code"].fillna("").astype(str).str.strip()
    has_code = subj.ne("") & subj.ne("nan")
    has_image = subj.isin(img_subjects)

    # "No" only when there IS a subject_code but the subject has no imaging.
    df["image_downloaded"] = "Yes"
    df.loc[has_code & ~has_image, "image_downloaded"] = "No"

    out = args.out or args.csv.with_name(f"{args.csv.stem}-image_downloaded.csv")
    df.to_csv(out, index=False)

    n_no = int((df["image_downloaded"] == "No").sum())
    n_yes = int((df["image_downloaded"] == "Yes").sum())
    print(f"Subjects with images on disk : {len(img_subjects)}")
    print(f"Rows -> image_downloaded=Yes : {n_yes}")
    print(f"Rows -> image_downloaded=No  : {n_no}")
    print(f"  (rows without subject_code, counted as Yes: {int((~has_code).sum())})")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
