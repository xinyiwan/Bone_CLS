"""Copy each subset subject's full data folder (images + segmentations +
review/assessment.json) from the sorted-data tree into a local "subset" dir.

Run locally, on the machine with /Volumes/SanDisk/BONE-AI/tmp_sorted_data
(or wherever --data-root points) mounted.
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd

DEFAULT_DATA_ROOT = Path("/Volumes/SanDisk/BONE-AI/tmp_sorted_data")
DEFAULT_OUT_DIR = Path("/Users/xinyi/Documents/github/Bone_CLS/output/subset_data/20pct")
DEFAULT_CLINICAL_CSV = Path("/Users/xinyi/Documents/github/Bone_CLS/kira-0515-seg.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("subset_csv", type=Path, help="CSV with a subject_code column")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--clinical-csv", type=Path, default=DEFAULT_CLINICAL_CSV,
                    help="clinical CSV to filter down to the subset's subjects")
    args = ap.parse_args()

    subjects = pd.read_csv(args.subset_csv)["subject_code"].astype(str).str.strip().tolist()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    clinical = pd.read_csv(args.clinical_csv)
    clinical["subject_code"] = clinical["subject_code"].astype(str).str.strip()
    filtered = clinical[clinical["subject_code"].isin(subjects)]
    filtered_out = args.out_dir / args.clinical_csv.name
    filtered.to_csv(filtered_out, index=False)
    print(f"Filtered clinical CSV: {len(filtered)} rows -> {filtered_out}")

    copied, skipped, missing = 0, 0, []
    for subject in subjects:
        src = args.data_root / subject
        dst = args.out_dir / subject
        if not src.is_dir():
            missing.append(subject)
            continue
        if dst.exists():
            skipped += 1
            continue
        shutil.copytree(src, dst)
        copied += 1

    print(f"Copied  : {copied}")
    print(f"Skipped (already present): {skipped}")
    if missing:
        print(f"Missing on disk ({len(missing)}): {', '.join(missing)}")
    print(f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
