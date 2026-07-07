"""Flag which patient ids from the CSV have images downloaded locally.

Folder layout of the nifti root (four download batches, each with subjects)::

    nifti/
      <batch>/
        BONE_AI_XX/
          <session>/
            <scan>/
              images.nii

A patient counts as *downloaded* if at least one image file (default
``images.nii``) exists anywhere under a ``BONE_AI_<n>`` folder. The subject id
is read straight from the path with a ``BONE_AI_\\d+`` regex, so the exact nesting
depth / batch-folder names don't matter.

Outputs:
  - ``<csv stem>-downloaded.csv`` : the original CSV plus a ``downloaded`` column
    ("Yes"/"No"; empty when the row has no patient id).
  - ``unavailable_subjects.csv``  : the unique patient ids present in the CSV but
    with NO images downloaded (the "unavailable" set).
"""

import argparse
import re
from pathlib import Path

import pandas as pd

DEFAULT_CSV = Path("/output/kira-0515-seg.csv")
DEFAULT_NIFTI_ROOT = Path("/data")
DEFAULT_ID_COL = "subject_code"
DEFAULT_IMAGE_GLOB = "images.nii"

PID_RE = re.compile(r"(BONE_AI_\d+)")


def downloaded_subjects(nifti_root: Path, image_glob: str) -> set:
    """Set of BONE_AI_* ids that have at least one image file under nifti_root."""
    subjects = set()
    for img in nifti_root.rglob(image_glob):
        m = PID_RE.search(str(img))
        if m:
            subjects.add(m.group(1))
    return subjects


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("csv", nargs="?", type=Path, default=DEFAULT_CSV,
                    help=f"source CSV with patient ids (default: {DEFAULT_CSV})")
    ap.add_argument("--nifti-root", type=Path, default=DEFAULT_NIFTI_ROOT,
                    help=f"root of the nifti download tree (default: {DEFAULT_NIFTI_ROOT})")
    ap.add_argument("--id-col", default=DEFAULT_ID_COL,
                    help=f"patient-id column in the CSV (default: {DEFAULT_ID_COL})")
    ap.add_argument("--image-glob", default=DEFAULT_IMAGE_GLOB,
                    help=f"image filename to look for (default: {DEFAULT_IMAGE_GLOB})")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output CSV (default: <csv stem>-downloaded.csv)")
    args = ap.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")
    if not args.nifti_root.exists():
        raise SystemExit(
            f"nifti root not found: {args.nifti_root}\n"
            "Run on the machine that has the data, or pass --nifti-root."
        )

    df = pd.read_csv(args.csv)
    if args.id_col not in df.columns:
        raise SystemExit(f"'{args.id_col}' column not in {args.csv.name}")

    have = downloaded_subjects(args.nifti_root, args.image_glob)
    # fillna("") first: read_csv may use the pandas 'str' dtype, where NaN stays
    # a float and astype(str) would NOT turn it into "nan".
    pid = df[args.id_col].fillna("").astype(str).str.strip()
    has_id = pid.ne("") & pid.ne("nan")
    is_down = pid.isin(have)

    df["downloaded"] = ""
    df.loc[has_id & is_down, "downloaded"] = "Yes"
    df.loc[has_id & ~is_down, "downloaded"] = "No"

    out = args.out or args.csv.with_name(f"{args.csv.stem}-downloaded.csv")
    df.to_csv(out, index=False)

    csv_subjects = set(pid[has_id])
    unavailable = sorted(csv_subjects - have)
    unavail_out = out.with_name("unavailable_subjects.csv")
    pd.DataFrame({args.id_col: unavailable}).to_csv(unavail_out, index=False)

    print(f"Subjects with images downloaded (under {args.nifti_root}): {len(have)}")
    print(f"Unique patient ids in CSV                              : {len(csv_subjects)}")
    print(f"  downloaded (available)   : {len(csv_subjects) - len(unavailable)}")
    print(f"  NOT downloaded (unavailable): {len(unavailable)}")
    print(f"Rows -> downloaded=Yes: {int((df['downloaded'] == 'Yes').sum())} | "
          f"No: {int((df['downloaded'] == 'No').sum())} | "
          f"blank (no id): {int((~has_id).sum())}")
    print(f"\nSaved -> {out}")
    print(f"Saved -> {unavail_out}")


if __name__ == "__main__":
    main()
