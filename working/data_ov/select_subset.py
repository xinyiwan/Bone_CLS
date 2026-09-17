"""Select a stratified subset of subjects for the 150-subject cohort.

Pool = subjects with imaging on disk ∩ subjects in the clinical CSV, minus
subjects with If_segmented == "exclude" (see distribution_imaging.py for the
same imaging/CSV overlap logic).

Subset size is either an absolute --n or a --pct of the pool (e.g. --pct 20
for 20%); pass exactly one. Allocate that many subjects across groups
proportional to each group's share of the pool (largest-remainder rounding
so the total is exact; every group with at least --min-per-group pool
subjects gets at least --min-per-group slots, so rare groups aren't dropped
entirely). Groups are either diagnosis subtypes (--group-by subtype,
default) or the coarser benign/malignant/mimicker/none/uncertain categories
from lesion_category.py (--group-by category). Within each group, prefer
subjects that already have a feature-label JSON
(<subject>/*/review/*/assessment.json on disk) and/or a segmentation
(If_segmented == "done", or a segmentation_history/segs folder on disk),
highest priority first.

Run on the machine that holds the imaging data:
    python select_subset.py [csv] --data-root /path/to/tmp_sorted_data \
        --pct 20 --group-by category --min-per-group 2 --out subset.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from distribution_imaging import subject_labels, subjects_with_images
from lesion_category import to_category

DEFAULT_CSV = Path("/Users/xinyi/Documents/github/Bone_CLS/kira-0515-seg.csv")
DEFAULT_DATA_ROOT = Path("/Volumes/SanDisk/BONE-AI/tmp_sorted_data")
OUT_CSV = Path("/Users/xinyi/Documents/github/Bone_CLS/output/subset/subset_20pct.csv")
SEG_HISTORY_DIRNAME = "segmentation_history"
SEG_SUBDIR = "segs"
SEG_SUFFIXES = ("_seg.nii.gz", "_seg.nii")
REVIEW_DIRNAME = "review"
ASSESSMENT_NAME = "assessment.json"


def subjects_with_seg_history(data_root: Path) -> set:
    """Subjects with a <subject>/*/segmentation_history/segs/*_seg.nii(.gz) on disk."""
    subjects = set()
    for segs_dir in data_root.glob(f"*/*/{SEG_HISTORY_DIRNAME}/{SEG_SUBDIR}"):
        if any(p.name.endswith(SEG_SUFFIXES) for p in segs_dir.iterdir()):
            subjects.add(segs_dir.parents[2].name)
    return subjects


def subjects_with_feature_json(data_root: Path) -> set:
    """Subjects with a <subject>/*/review/*/assessment.json on disk."""
    subjects = set()
    for json_path in data_root.glob(f"*/*/{REVIEW_DIRNAME}/*/{ASSESSMENT_NAME}"):
        subjects.add(json_path.parents[3].name)
    return subjects


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("csv", nargs="?", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--group-by", choices=("subtype", "category"), default="subtype",
                    help="stratify by diagnosis subtype (default) or coarse category")
    size = ap.add_mutually_exclusive_group()
    size.add_argument("--n", type=int, default=None, help="absolute subset size")
    size.add_argument("--pct", type=float, default=None,
                      help="subset size as a percentage of the pool, e.g. 20 for 20%%")
    ap.add_argument("--min-per-group", type=int, default=1,
                    help="minimum slots for any group that has at least that many pool subjects")
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    args = ap.parse_args()

    clinical = pd.read_csv(args.csv)
    clinical["subject_code"] = clinical["subject_code"].astype(str).str.strip()
    excluded = set(
        clinical.loc[clinical["If_segmented"].astype(str).str.strip() == "exclude",
                     "subject_code"]
    )
    segmented = set(
        clinical.loc[clinical["If_segmented"].astype(str).str.strip() == "done",
                     "subject_code"]
    ) | subjects_with_seg_history(args.data_root)

    per_subject = subject_labels(args.csv)
    img_subjects = subjects_with_images(args.data_root)
    labeled = subjects_with_feature_json(args.data_root)

    pool = sorted((set(per_subject.index) & img_subjects) - excluded)
    if not pool:
        raise SystemExit("Empty pool -- check --data-root / csv paths.")

    df = pd.DataFrame({"subject_code": pool})
    df["label"] = df["subject_code"].map(per_subject)
    df["category"] = df["label"].map(to_category)
    df["has_feature_json"] = df["subject_code"].isin(labeled)
    df["has_seg"] = df["subject_code"].isin(segmented)
    df["priority"] = df["has_feature_json"].astype(int) + df["has_seg"].astype(int)

    group_col = "label" if args.group_by == "subtype" else "category"

    n_requested = args.n if args.pct is None else round(args.pct / 100 * len(df))
    if n_requested is None:
        raise SystemExit("Pass either --n or --pct.")

    # Proportional allocation per group, largest-remainder rounding to hit n
    # exactly, with every eligible group guaranteed at least --min-per-group
    # (capped by its pool size).
    min_per_group = args.min_per_group
    group_sizes = df[group_col].value_counts()
    n = min(n_requested, len(df))
    raw = group_sizes / len(df) * n
    floor = group_sizes.clip(upper=min_per_group)  # can't floor above what a group has
    alloc = raw.astype(int).clip(lower=floor)
    alloc = alloc.clip(upper=group_sizes)

    diff = int(n - alloc.sum())
    if diff > 0:
        # Give the extras to groups with the largest unrounded remainder that still have room.
        room = (group_sizes - alloc).sort_values(ascending=False)
        for label in room.index:
            if diff == 0:
                break
            give = min(diff, int(room[label]))
            alloc[label] += give
            diff -= give
    elif diff < 0:
        # Too many forced minimums -- claw back from the groups that can spare it, largest first.
        spare = (alloc - floor).sort_values(ascending=False)
        for label in spare.index:
            if diff == 0:
                break
            take = min(-diff, int(spare[label]))
            alloc[label] -= take
            diff += take

    picked = []
    for group_value, k in alloc.items():
        if k <= 0:
            continue
        group = df[df[group_col] == group_value].sort_values(
            ["priority", "subject_code"], ascending=[False, True]
        )
        picked.append(group.head(k))
    subset = pd.concat(picked, ignore_index=True)

    # Redistribute any shortfall (small groups exhausted) into the largest remaining group.
    shortfall = n - len(subset)
    if shortfall > 0:
        remaining = df[~df["subject_code"].isin(subset["subject_code"])].sort_values(
            ["priority", "subject_code"], ascending=[False, True]
        )
        subset = pd.concat([subset, remaining.head(shortfall)], ignore_index=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    subset.sort_values(["label", "subject_code"]).to_csv(args.out, index=False)

    print(f"Grouped by                : {group_col}")
    print(f"Pool size                 : {len(df)}")
    print(f"Selected                  : {len(subset)}")
    print(f"  with feature JSON       : {subset['has_feature_json'].sum()}")
    print(f"  with segmentation       : {subset['has_seg'].sum()}")
    print(f"\nPer-{group_col}: pool vs selected")
    cmp = pd.DataFrame({
        "pool": group_sizes,
        "selected": subset[group_col].value_counts(),
    }).fillna(0).astype(int)
    print(cmp.to_string())
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
