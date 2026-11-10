"""Select a subset via even allocation across diagnosis subtypes, instead of
select_subset.py's proportional-to-pool-size allocation.

Pool = subjects with imaging on disk ∩ subjects in the clinical CSV, minus
subjects with If_segmented == "exclude" (same as select_subset.py) and minus
subjects whose diagnosis label is "uncertain by reports" / missing.

With k subtypes (the same subtype labels as distribution.py) and target
T = round(--pct/100 * pool_size), an even split gives each subtype a fixed
share n_k = T/k. Subtypes with fewer than n_k pool subjects contribute ALL
of their subjects (set a); subtypes with n_k or more contribute exactly
n_k (set b) -- the share is NOT re-split after set a is removed, so the
actual total is generally smaller than T (every undersized subtype leaves
its slack on the table). That gap is expected and not corrected for: this
picks share = n_k directly from --pct, no search over p, because the gap is
provably monotonic in p (smaller p is always closer to its own target), so
"searching for the closest p" has no non-trivial answer.

Set a (undersized subtypes) is taken in full, no ordering needed. Set b
(subtypes capped at their share) prefers subjects with a segmentation
(If_segmented == "done", or a segmentation_history/segs folder on disk).

Run on the machine that holds the imaging data:
    python select_subset_waterfill.py [csv] --data-root /path/to/tmp_sorted_data \
        --pct 20 --out subset_waterfill.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from distribution_imaging import subject_labels, subjects_with_images
from labels import MISSING_LABEL, UNCERTAIN_LABEL
from select_subset import subjects_with_seg_history

DEFAULT_CSV = Path("/Users/xinyi/Documents/github/Bone_CLS/output/merged_data/kira-1104-combined_dataset.filtered.csv")
DEFAULT_DATA_ROOT = Path("/Volumes/SanDisk/BONE-AI/tmp_sorted_data")
OUT_CSV = Path("/Users/xinyi/Documents/github/Bone_CLS/output/subset/subset_waterfill_filtered1106-15pct.csv")


def even_split_alloc(group_sizes: pd.Series, target: int) -> pd.Series:
    """Fixed even share n_k = target/k: undersized groups take everything
    (set a), oversized groups take exactly n_k (set b) -- no redistribution
    of set a's slack, so sum(alloc) can fall short of target."""
    k = len(group_sizes)
    share = target / k
    undersized = group_sizes <= share
    alloc = group_sizes.where(undersized, int(share))
    return alloc.astype(int)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("csv", nargs="?", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--pct", type=float, required=True,
                    help="target subset size as a percentage of the pool, e.g. 20 for 20%%")
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
    # uncertain / missing diagnoses are not a subtype: leave them out of the pool
    n_unc = per_subject.isin([UNCERTAIN_LABEL, MISSING_LABEL]).sum()
    per_subject = per_subject[~per_subject.isin([UNCERTAIN_LABEL, MISSING_LABEL])]
    print(f"Left out (uncertain/missing label): {n_unc} subjects")
    img_subjects = subjects_with_images(args.data_root)

    pool = sorted((set(per_subject.index) & img_subjects) - excluded)
    if not pool:
        raise SystemExit("Empty pool -- check --data-root / csv paths.")

    df = pd.DataFrame({"subject_code": pool})
    df["label"] = df["subject_code"].map(per_subject)
    df["has_seg"] = df["subject_code"].isin(segmented)

    group_sizes = df["label"].value_counts()
    target = round(args.pct / 100 * len(df))
    alloc = even_split_alloc(group_sizes, target)

    picked = []
    for label, k in alloc.items():
        if k <= 0:
            continue
        group = df[df["label"] == label].sort_values(
            ["has_seg", "subject_code"], ascending=[False, True]
        )
        picked.append(group.head(k))
    subset = pd.concat(picked, ignore_index=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    subset.sort_values(["label", "subject_code"]).to_csv(args.out, index=False)

    n_full = (alloc == group_sizes.reindex(alloc.index)).sum()
    print(f"Pool size                 : {len(df)}")
    print(f"Target (pct * pool)       : {target}")
    print(f"Selected                  : {len(subset)}  (gap: {len(subset) - target})")
    print(f"  with segmentation       : {subset['has_seg'].sum()}")
    print(f"Subtypes taken in full (set a): {n_full}/{len(alloc)}")
    print("\nPer-subtype: pool vs selected")
    cmp = pd.DataFrame({"pool": group_sizes, "selected": alloc}).fillna(0).astype(int)
    print(cmp.to_string())
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
