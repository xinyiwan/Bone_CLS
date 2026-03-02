"""
Visualise sequence-type distribution from a labelled DICOM header CSV.

Produces a single figure with two horizontal bar charts:
  left  – number of unique subjects per modality combination
  right – total number of images (slices) per modality combination

Modality combination = sequence_type × fat_sat × contrast
  e.g. "T2 FS", "T1 Gd", "DWI", "T1 FS Gd"

Usage:
    python visualise_labels.py labelled.csv          # saves alongside CSV
    python visualise_labels.py labelled.csv out.png  # custom output path
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def build_modality_label(row: pd.Series) -> str:
    parts = [row["sequence_type"]]
    if row["fat_sat"]:
        parts.append("FS")
    if row["contrast"]:
        parts.append("CE")
    return " ".join(parts)


def visualise(input_csv: Path, output_fig: Path) -> None:
    df = pd.read_csv(input_csv, dtype=str).fillna("")

    # Drop rows without a sequence type
    df = df[df["sequence_type"].str.strip() != ""]
    print(f"{len(df)} labelled scans retained")

    df["modality"] = df.apply(build_modality_label, axis=1)

    # Subject count: unique subjects per modality combo
    subj_counts = (
        df.groupby("modality")["subject"]
        .nunique()
        .sort_values(ascending=False)
    )

    # Image count: rows per modality combo (one row = one image)
    img_counts = df.groupby("modality").size().reindex(subj_counts.index)

    n = len(subj_counts)
    labels = subj_counts.index.tolist()
    y = range(n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(4, n * 0.45)))

    # --- subjects ---
    bars1 = ax1.barh(y, subj_counts.values, color="steelblue")
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel("Number of subjects")
    ax1.set_title("Subjects per modality")
    ax1.invert_yaxis()
    for bar, val in zip(bars1, subj_counts.values):
        ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                 str(val), va="center", fontsize=8)

    # --- images ---
    bars2 = ax2.barh(y, img_counts.values, color="coral")
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("Number of images")
    ax2.set_title("Images per modality")
    ax2.invert_yaxis()
    for bar, val in zip(bars2, img_counts.values):
        ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                 str(int(val)), va="center", fontsize=8)

    plt.suptitle("MRI sequence distribution", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(output_fig, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {output_fig}")

    # Text summary
    print("\nSubjects per modality:")
    print(subj_counts.to_string())
    print("\nImages per modality:")
    print(img_counts.to_string())


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        sys.exit("Usage: python visualise_labels.py labelled.csv [output.png]")
    input_csv  = Path(args[0])
    output_fig = Path(args[1]) if len(args) > 1 else input_csv.with_suffix(".png")
    visualise(input_csv, output_fig)
