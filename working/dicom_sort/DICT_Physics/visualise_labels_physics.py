"""
Visualise physics-label distribution from a labelled DICOM header CSV
(output of label_physics.py).

Figure layout (2 × 2):
  top-left    – subjects per modality (all)
  top-right   – images per modality (all)
  bottom-left – subjects per modality (top 6, excl. Localizer / Unknown)
  bottom-right– images stacked by acquisition type (FSE / SE / GRE / IR / FLAIR)
                for the same top 6

Modality label = phys_sequence [+ fat_sat] [+ Contrast]
  e.g. "T2W-STIR", "T1W-FS-Contrast", "DWI"

Usage:
    python visualise_labels_physics.py labelled.csv          # saves alongside CSV
    python visualise_labels_physics.py labelled.csv out.png  # custom output path
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


ACQ_ORDER  = ["FSE", "SE", "GRE", "IR", "FLAIR", "Other"]
ACQ_COLOR  = {
    "FSE":   "#4e79a7",
    "SE":    "#76b7b2",
    "GRE":   "#f28e2b",
    "IR":    "#e15759",
    "FLAIR": "#b07aa1",
    "Other": "#bab0ac",
}

EXCLUDE_SEQ = {"Localizer", "Unknown", "Unknown_GRE", "UNKNOWN", ""}


def build_modality_label(row: pd.Series) -> str:
    """Combine phys_sequence + phys_fat_sat + phys_contrast into a display label."""
    parts = [str(row.get("phys_sequence", "")).strip()]
    fs = str(row.get("phys_fat_sat", "")).strip()
    c  = str(row.get("phys_contrast", "")).strip()
    if fs:
        parts.append(fs)       # "FS" or "STIR"
    if c == "Contrast":
        parts.append("Contrast")
    return "-".join(p for p in parts if p)


def _acq_group(val: str) -> str:
    v = str(val).strip()
    return v if v in ACQ_ORDER else "Other"


def _hbar(ax, y, values, labels, color, xlabel, title):
    bars = ax.barh(y, values, color=color)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.invert_yaxis()
    mx = max(values) if values else 1
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + mx * 0.01,
                bar.get_y() + bar.get_height() / 2,
                str(int(val)), va="center", fontsize=8)


def visualise(input_csv: Path, output_fig: Path) -> None:
    df = pd.read_csv(input_csv, dtype=str).fillna("")

    for col in ["phys_sequence", "phys_acquisition", "phys_fat_sat", "phys_contrast"]:
        if col not in df.columns:
            raise SystemExit(f"Column '{col}' not found — run label_physics.py first.")

    print(f"{len(df)} rows loaded")

    df["modality"] = df.apply(build_modality_label, axis=1)
    df["acq_group"] = df["phys_acquisition"].apply(_acq_group)

    # ── all modalities ────────────────────────────────────────────────────────
    all_subj = (df.groupby("modality")["subject"].nunique()
                  .sort_values(ascending=False))
    all_imgs = df.groupby("modality").size().reindex(all_subj.index)

    # ── top 6: exclude Localizer / Unknown variants ───────────────────────────
    excl_mask = df["phys_sequence"].isin(EXCLUDE_SEQ)
    top6 = (df[~excl_mask]
            .groupby("modality")["subject"].nunique()
            .sort_values(ascending=False)
            .head(6))
    df_top = df[df["modality"].isin(top6.index)]

    acq_counts = (
        df_top.groupby(["modality", "acq_group"])
        .size()
        .unstack(fill_value=0)
        .reindex(top6.index)
        .fillna(0)
    )
    acq_cols = [c for c in ACQ_ORDER if c in acq_counts.columns]
    acq_counts = acq_counts[acq_cols]

    # ── layout ────────────────────────────────────────────────────────────────
    n_all = len(all_subj)
    n_top = len(top6)
    h_all = max(3, n_all * 0.4)
    h_top = max(2, n_top * 0.55)

    fig = plt.figure(figsize=(16, h_all + h_top + 1.5))
    gs  = gridspec.GridSpec(2, 2,
                            height_ratios=[h_all, h_top],
                            hspace=0.5, wspace=0.4)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    _hbar(ax1, list(range(n_all)), all_subj.values.tolist(),
          all_subj.index.tolist(), "steelblue",
          "Number of subjects", "Subjects per modality (all)")

    _hbar(ax2, list(range(n_all)), all_imgs.values.tolist(),
          all_subj.index.tolist(), "coral",
          "Number of images", "Images per modality (all)")

    _hbar(ax3, list(range(n_top)), top6.values.tolist(),
          top6.index.tolist(), "steelblue",
          "Number of subjects", "Subjects — top 6")

    # bottom-right: stacked acquisition breakdown for top 6
    left = [0] * n_top
    for col in acq_cols:
        vals = acq_counts[col].values.tolist()
        bars = ax4.barh(list(range(n_top)), vals, left=left,
                        label=col, color=ACQ_COLOR.get(col, "#bab0ac"))
        for bar, v, l in zip(bars, vals, left):
            if v > 0:
                ax4.text(l + v / 2, bar.get_y() + bar.get_height() / 2,
                         str(int(v)), ha="center", va="center",
                         fontsize=7, color="white", fontweight="bold")
        left = [l + v for l, v in zip(left, vals)]

    ax4.set_yticks(list(range(n_top)))
    ax4.set_yticklabels(top6.index.tolist())
    ax4.set_xlabel("Number of images")
    ax4.set_title("Images by acquisition type — top 6")
    ax4.invert_yaxis()
    ax4.legend(loc="lower right", fontsize=8)

    plt.suptitle("Physics-label distribution", fontsize=14, y=1.01)
    plt.savefig(output_fig, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {output_fig}")

    print("\nSubjects per modality:")
    print(all_subj.to_string())
    print("\nImages per modality:")
    print(all_imgs.to_string())
    print("\nImages by acquisition type (top 6):")
    print(acq_counts.to_string())


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        sys.exit("Usage: python visualise_labels_physics.py labelled.csv [output.png]")
    input_csv  = Path(args[0])
    output_fig = Path(args[1]) if len(args) > 1 else input_csv.with_suffix(".png")
    visualise(input_csv, output_fig)
