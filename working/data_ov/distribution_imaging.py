"""Diagnosis distribution restricted to subjects that have imaging on disk.

Unlike ``distribution.py`` (which counts every report row), this script:
  1. Walks the sorted-data tree for ``<root>/<subject>/.../images.nii.gz`` and
     collects the set of subjects that actually have at least one image.
  2. Reads the segmentation record CSV (has both ``subject_code`` and
     ``palabra_manual``) and derives ONE diagnosis label PER UNIQUE SUBJECT:
        - each row's palabra_manual -> label via labels.to_label
        - if a subject's rows disagree on the final label -> "uncertain by reports"
  3. Keeps only subjects present in BOTH sets (imaging ∩ CSV) and plots the
     per-subject diagnosis distribution.

Run this on the machine that holds the data (default root is the remote path).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from labels import UNCERTAIN_LABEL, to_label

DEFAULT_CSV =  Path("/output/kira-0515-seg.csv")
DEFAULT_DATA_ROOT = Path("/data")
OUT_DIR = Path("/output/clinical_info")
IMAGE_NAME = "images.nii.gz"
COLUMN = "palabra_manual"


def subjects_with_images(data_root: Path) -> set:
    """Top-level subject folders under data_root that contain any images.nii.gz."""
    subjects = set()
    for img in data_root.rglob(IMAGE_NAME):
        rel = img.relative_to(data_root).parts
        if rel:
            subjects.add(rel[1])
    return subjects


def subject_labels(csv_path: Path) -> pd.Series:
    """One diagnosis label per subject_code; disagreeing rows -> uncertain."""
    df = pd.read_csv(csv_path)
    df["subject_code"] = df["subject_code"].astype(str).str.strip()
    df = df[df["subject_code"].ne("") & df["subject_code"].ne("nan")]
    df["_label"] = df[COLUMN].map(to_label)

    def collapse(labels: pd.Series) -> str:
        uniq = set(labels)
        return next(iter(uniq)) if len(uniq) == 1 else UNCERTAIN_LABEL

    return df.groupby("subject_code")["_label"].apply(collapse)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("csv", nargs="?", type=Path, default=DEFAULT_CSV,
                    help=f"segmentation record CSV (default: {DEFAULT_CSV})")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                    help="root of the BONE_AI_* data tree to scan for images")
    args = ap.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")
    if not args.data_root.exists():
        raise SystemExit(
            f"Data root not found: {args.data_root}\n"
            "Run on the machine that has the data, or pass --data-root."
        )

    per_subject = subject_labels(args.csv)
    csv_subjects = set(per_subject.index)
    img_subjects = subjects_with_images(args.data_root)
    overlap = csv_subjects & img_subjects

    print(f"Subjects in CSV                : {len(csv_subjects)}")
    print(f"Subjects with images on disk   : {len(img_subjects)}")
    print(f"Overlap (imaging ∩ CSV)        : {len(overlap)}")
    missing_img = sorted(csv_subjects - img_subjects)
    if missing_img:
        print(f"In CSV but NO image on disk    : {len(missing_img)} "
              f"(e.g. {', '.join(missing_img[:10])})")
    no_csv = sorted(img_subjects - csv_subjects)
    if no_csv:
        print(f"Image on disk but NOT in CSV   : {len(no_csv)} "
              f"(e.g. {', '.join(no_csv[:10])})")

    counts = per_subject.loc[sorted(overlap)].value_counts()
    total = int(counts.sum())
    print(f"\nSubjects plotted               : {total}\n")

    dist = pd.DataFrame({
        "subtype": counts.index,
        "count": counts.values,
        "percent": (counts.values / total * 100).round(2),
    })

    csv_out = OUT_DIR / "distribution_imaging.csv"
    dist.to_csv(csv_out, index=False)
    print(f"Saved table -> {csv_out}")
    print("\nDistribution (per subject with imaging):")
    print(dist.to_string(index=False))

    fig, ax = plt.subplots(figsize=(12, max(6, 0.3 * len(counts))))
    ax.barh(dist["subtype"][::-1], dist["count"][::-1], color="seagreen")
    ax.set_xlabel("Number of subjects")
    ax.set_ylabel("Diagnosis (English)")
    ax.set_title(f"Diagnosis distribution — subjects with imaging (n={total})")
    for i, (c, p) in enumerate(zip(dist["count"][::-1], dist["percent"][::-1])):
        ax.text(c, i, f" {c} ({p}%)", va="center", fontsize=8)
    plt.tight_layout()

    png_out = OUT_DIR / "distribution_imaging.png"
    plt.savefig(png_out, dpi=150)
    print(f"\nSaved plot  -> {png_out}")


if __name__ == "__main__":
    main()
