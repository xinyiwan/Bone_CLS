"""
Visualise sequence-type distribution from a labelled DICOM header CSV.

Figure layout (2 × 2):
  top-left    – subjects per modality (all)
  top-right   – images per modality (all)
  bottom-left – subjects per modality (top 6, excl. localizer)
  bottom-right– images stacked by orientation (top 6, with counts)

Modality combination = sequence_type × fat_sat × contrast
  e.g. "T2-FS", "T1-CE", "DWI", "T1-FS-CE"

Usage:
    python visualise_labels.py labelled.csv          # saves alongside CSV
    python visualise_labels.py labelled.csv out.png  # custom output path
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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


def _hbar(ax, y, values, labels, color, xlabel, title):
    """Draw a plain horizontal bar chart with value labels."""
    bars = ax.barh(y, values, color=color)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.invert_yaxis()
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                str(int(val)), va="center", fontsize=8)


def visualise(input_csv: Path, output_fig: Path) -> None:
    df = pd.read_csv(input_csv, dtype=str).fillna("")

    df = df[df["sequence_type"].str.strip() != ""]
    print(f"{len(df)} labelled rows retained")

    df["modality"]         = df.apply(build_modality_label, axis=1)
    df["orientation_type"] = pd.to_numeric(df["orientation_type"], errors="coerce")
    df["orient_label"]     = df["orientation_type"].map(ORIENT_MAP).fillna("Other")

    # ── all modalities ────────────────────────────────────────────────────────
    all_subj = (df.groupby("modality")["subject"].nunique()
                  .sort_values(ascending=False))
    all_imgs = df.groupby("modality").size().reindex(all_subj.index)

    # ── top 6 non-localizer ───────────────────────────────────────────────────
    top6 = all_subj[~all_subj.index.str.contains("localizer", case=False)].head(6)
    df_top = df[df["modality"].isin(top6.index)]

    orient_counts = (
        df_top.groupby(["modality", "orient_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(top6.index)
        .fillna(0)
    )
    orient_cols   = [c for c in ORIENT_ORDER if c in orient_counts.columns]
    orient_counts = orient_counts[orient_cols]

    # ── layout ────────────────────────────────────────────────────────────────
    n_all = len(all_subj)
    n_top = len(top6)
    h_all = max(3, n_all * 0.4)
    h_top = max(2, n_top * 0.55)

    fig = plt.figure(figsize=(16, h_all + h_top + 1.5))
    gs  = gridspec.GridSpec(2, 2, height_ratios=[h_all, h_top], hspace=0.5, wspace=0.4)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    y_all = list(range(n_all))
    y_top = list(range(n_top))

    # top-left: subjects (all)
    _hbar(ax1, y_all, all_subj.values, all_subj.index.tolist(),
          "steelblue", "Number of subjects", "Subjects per modality")

    # top-right: images (all)
    _hbar(ax2, y_all, all_imgs.values, all_subj.index.tolist(),
          "coral", "Number of images", "Images per modality")

    # bottom-left: subjects (top 6)
    _hbar(ax3, y_top, top6.values, top6.index.tolist(),
          "steelblue", "Number of subjects", "Subjects — top 6")

    # bottom-right: stacked orientation (top 6) with counts
    left = [0] * n_top
    for col in orient_cols:
        vals = orient_counts[col].values.tolist()
        bars = ax4.barh(y_top, vals, left=left, label=col, color=ORIENT_COLOR[col])
        # label each non-zero segment
        for bar, v, l in zip(bars, vals, left):
            if v > 0:
                ax4.text(l + v / 2, bar.get_y() + bar.get_height() / 2,
                         str(int(v)), ha="center", va="center",
                         fontsize=7, color="white", fontweight="bold")
        left = [l + v for l, v in zip(left, vals)]

    ax4.set_yticks(y_top)
    ax4.set_yticklabels(top6.index.tolist())
    ax4.set_xlabel("Number of images")
    ax4.set_title("Images by orientation — top 6")
    ax4.invert_yaxis()
    ax4.legend(loc="lower right", fontsize=8)

    plt.suptitle("MRI sequence distribution", fontsize=14, y=1.01)
    plt.savefig(output_fig, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {output_fig}")

    print("\nSubjects per modality:")
    print(all_subj.to_string())
    print("\nImages per modality:")
    print(all_imgs.to_string())
    print("\nImages by orientation (top 6):")
    print(orient_counts.to_string())


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        sys.exit("Usage: python visualise_labels.py labelled.csv [output.png]")
    input_csv  = Path(args[0])
    output_fig = Path(args[1]) if len(args) > 1 else input_csv.with_suffix(".png")
    visualise(input_csv, output_fig)
