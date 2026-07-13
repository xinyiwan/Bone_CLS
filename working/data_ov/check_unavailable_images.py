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

The main CSV's ``accession_number`` column is corrupted (Excel rounded the
16-digit ids into scientific notation, e.g. ``9.04E+15``). When a corrected
source CSV is available (``--accession-csv``), its accession numbers are joined
back in on ``info_key`` + ``sip`` and written to a new ``accession_number_corrected``
column, leaving the original column untouched.

Outputs:
  - ``<csv stem>-downloaded.csv`` : the original CSV plus a ``downloaded`` column
    ("Yes"/"No"; empty when the row has no patient id), and an
    ``accession_number_corrected`` column when a corrected source CSV is given.
  - ``unavailable_subjects.csv``  : the unique patient ids present in the CSV but
    with NO images downloaded (the "unavailable" set).
"""

import argparse
import fnmatch
import os
import re
from pathlib import Path

import pandas as pd

DEFAULT_CSV = Path("/output/kira-0515-seg.csv")
DEFAULT_NIFTI_ROOT = Path("/data")
DEFAULT_ID_COL = "subject_code"
DEFAULT_IMAGE_GLOB = "images.nii.gz"

# Source CSV that still holds the untruncated accession numbers, and the
# columns used to line its rows up with the main CSV.
DEFAULT_ACCESSION_CSV = None
ACCESSION_COL = "accession_number"
ACCESSION_KEYS = ["info_key", "sip"]
ACCESSION_OUT_COL = "accession_number_corrected"

PID_RE = re.compile(r"(BONE_AI_\d+)")


def downloaded_subjects(nifti_root: Path, image_glob: str) -> set:
    """Set of BONE_AI_* ids that have at least one image file under nifti_root.

    Uses os.walk with followlinks=True: the nifti tree lives on a network mount
    where directories are symlinks, and pathlib.rglob does NOT descend into
    symlinked directories (so it silently finds nothing).
    """
    subjects = set()
    for dirpath, _dirnames, filenames in os.walk(nifti_root, followlinks=True):
        if not fnmatch.filter(filenames, image_glob):
            continue
        m = PID_RE.search(dirpath)
        if m:
            subjects.add(m.group(1))
    return subjects


def merge_corrected_accession(df: pd.DataFrame, accession_csv: Path) -> int:
    """Add ACCESSION_OUT_COL to df from accession_csv, joined on ACCESSION_KEYS.

    The accession column is read (and kept) as raw TEXT, never as a number: the
    ids are 16 digits, above float64's exact-integer limit (2**53), so letting
    pandas parse them numerically rounds the last digit(s) to 0 -- the very
    corruption we are fixing, and it happens even via the nullable-Int64 path on
    some pandas versions. Reading as string copies the exact characters from the
    source CSV. Returns the number of rows that got a value.
    """
    src = pd.read_csv(accession_csv, dtype={ACCESSION_COL: "string"})
    missing = [c for c in ACCESSION_KEYS + [ACCESSION_COL] if c not in src.columns]
    if missing:
        raise SystemExit(
            f"{accession_csv.name} is missing column(s): {', '.join(missing)}"
        )

    acc = src[ACCESSION_COL].str.strip()
    # If the source itself was already saved through Excel/float, the digits are
    # gone before we ever see them -- flag it rather than emit rounded ids.
    corrupted = acc.dropna().str.contains(r"[eE.]", regex=True)
    if corrupted.any():
        raise SystemExit(
            f"accession_number in {accession_csv.name} looks pre-rounded "
            f"(e.g. {acc.dropna()[corrupted].iloc[0]!r}); need a source with the "
            "full integer digits as text."
        )
    src = src.assign(**{ACCESSION_COL: acc})

    lookup = src[ACCESSION_KEYS + [ACCESSION_COL]].drop_duplicates(ACCESSION_KEYS)
    lookup = lookup.rename(columns={ACCESSION_COL: ACCESSION_OUT_COL})
    merged = df.merge(lookup, on=ACCESSION_KEYS, how="left")
    df[ACCESSION_OUT_COL] = merged[ACCESSION_OUT_COL].values
    return int(df[ACCESSION_OUT_COL].notna().sum())


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
    ap.add_argument("--accession-csv", type=Path, default=DEFAULT_ACCESSION_CSV,
                    help="CSV with untruncated accession numbers; joined on "
                         f"{'+'.join(ACCESSION_KEYS)} into '{ACCESSION_OUT_COL}' "
                         "(default: none / skip)")
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

    n_corrected = None
    if args.accession_csv is not None:
        if not args.accession_csv.exists():
            raise SystemExit(f"accession CSV not found: {args.accession_csv}")
        missing_keys = [c for c in ACCESSION_KEYS if c not in df.columns]
        if missing_keys:
            raise SystemExit(
                f"join key(s) {', '.join(missing_keys)} not in {args.csv.name}"
            )
        n_corrected = merge_corrected_accession(df, args.accession_csv)

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
    if n_corrected is not None:
        print(f"Corrected accession numbers ('{ACCESSION_OUT_COL}')       : "
              f"{n_corrected} / {len(df)} rows matched in {args.accession_csv.name}")
    print(f"\nSaved -> {out}")
    print(f"Saved -> {unavail_out}")


if __name__ == "__main__":
    main()
