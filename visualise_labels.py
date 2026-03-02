"""
Visualise sequence-type distribution from a labelled DICOM header CSV.

Produces a single figure with two horizontal bar charts:
  left  – number of unique subjects per modality (top 6, excl. localizer)
  right – image count stacked by orientation (axial / coronal / sagittal / other)

Modality combination = sequence_type × fat_sat × contrast
  e.g. "T2-FS", "T1-CE", "DWI", "T1-FS-CE"

Usage:
    python visualise_labels.py labelled.csv          # saves alongside CSV
    python visualise_labels.py labelled.csv out.png  # custom output path
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ORIENT_MAP   = {1: "Axial", 2: "Coronal", 3: "Sagittal"}
ORIENT_ORDER = ["Axial", "Coronal", "Sagittal", "Other"]
ORIENT_COLOR = {"Axial": "#4e79a7", "Coronal": "#f28e2b",
                "Sagittal": "#59a14f", "Other": "#bab0ac"}


def build_modality_label(row: pd.Series) -> str:
    parts = [row["sequence_type"]]
    if row["fat_sat"]:
        parts.append("FS")
    if row["contrast"]:
        parts.append("CE")
    return "-".join(parts)


def visualise(input_csv: Path, output_fig: Path) -> None:
    df = pd.read_csv(input_csv, dtype=str).fillna("")

    # Drop rows without a sequence type
    df = df[df["sequence_type"].str.strip() != ""]
    print(f"{len(df)} labelled rows retained")

    df["modality"]         = df.apply(build_modality_label, axis=1)
    df["orientation_type"] = pd.to_numeric(df["orientation_type"], errors="coerce")
    df["orient_label"]     = df["orientation_type"].map(ORIENT_MAP).fillna("Other")

    # All modalities – subject count, sorted
    all_subj = (
        df.groupby("modality")["subject"]
        .nunique()
        .sort_values(ascending=False)
    )

    # Top 6 non-localizer modalities
    top6 = all_subj[~all_subj.index.str.contains("localizer", case=False)].head(6)

    df_top = df[df["modality"].isin(top6.index)]

    # Image count by modality × orientation
    orient_counts = (
        df_top.groupby(["modality", "orient_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(top6.index)          # keep top-6 order
        .fillna(0)
    )
    # Only keep columns that exist, in canonical order
    orient_cols = [c for c in ORIENT_ORDER if c in orient_counts.columns]
    orient_counts = orient_counts[orient_cols]

    n      = len(top6)
    labels = top6.index.tolist()
    y      = list(range(n))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(4, n * 0.7)))

    # --- left: subjects ---
    bars1 = ax1.barh(y, top6.values, color="steelblue")
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel("Number of subjects")
    ax1.set_title("Subjects per modality (top 6)")
    ax1.invert_yaxis()
    for bar, val in zip(bars1, top6.values):
        ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                 str(val), va="center", fontsize=8)

    # --- right: stacked orientation ---
    left = [0] * n
    for col in orient_cols:
        vals = orient_counts[col].values
        ax2.barh(y, vals, left=left, label=col, color=ORIENT_COLOR[col])
        left = [l + v for l, v in zip(left, vals)]

    ax2.set_yticks(y)
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("Number of images")
    ax2.set_title("Images by orientation (top 6)")
    ax2.invert_yaxis()
    ax2.legend(loc="lower right", fontsize=8)

    plt.suptitle("MRI sequence distribution", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(output_fig, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {output_fig}")

    print("\nSubjects per modality (top 6):")
    print(top6.to_string())
    print("\nImages by orientation (top 6):")
    print(orient_counts.to_string())


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        sys.exit("Usage: python visualise_labels.py labelled.csv [output.png]")
    input_csv  = Path(args[0])
    output_fig = Path(args[1]) if len(args) > 1 else input_csv.with_suffix(".png")
    visualise(input_csv, output_fig)
