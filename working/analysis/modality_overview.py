"""
Imaging-modality distribution overview of the ground-truth review data.

Unlike clf_performance_analysis.py (which scores predictions against truth),
this script just describes the *ground truth*: what imaging modalities are
present and how often, across the reviewed series.

The modality of a series is the combination of three human-reviewed columns:

    Clase W Final   (weighting)      : T1W | T2W | DW | Other | T2* | PD | ...
    Clase FS Final  (fat suppression): Y | N | Y-STIR | -
    Clase C Final   (contrast)       : Y | N | -

joined with '-' into a single label, e.g.  T2W-Y-N  or  T1W-N-Y.

Normalisation (so equivalent modalities collapse together):
    * FS: 'Y-STIR' is folded into 'Y'.
    * C : contrast only applies to T1W. For every other weighting the C
          dimension is meaningless, so its '-' / 'N' variants are unified to
          'N'. This makes  T2W-Y--  identical to  T2W-Y-N , as requested.

Pre-processing mirrors clf_performance_analysis.py:
    * The input CSVs have 'Paciente' and 'Serie' swapped in their headers;
      we swap them back.
    * The CSVs (two or more) may overlap; they are concatenated and
      de-duplicated on the 'Nombre DICOM' path.
No truth buckets are dropped here — this is a full overview of everything.

Outputs (into --out-dir):
    modality_distribution.csv          full normalised modality counts
    modality_distribution_raw.csv      pre-normalisation counts (for diffing)
    modality_distribution.png          horizontal bar chart of the above
    weighting_distribution.{csv,png}   marginal distribution of Clase W Final
    fs_distribution.csv                marginal FS distribution
    c_distribution.csv                 marginal C distribution

Usage:
    python modality_overview.py \
        /Users/xinyi/Documents/github/Bone_CLS/Review_Sequence_Classifier.csv \
        /Users/xinyi/Documents/github/Bone_CLS/Review_Sequence_Classifier_n.csv \
        [more_review_csvs ...] \
        --out-dir /Users/xinyi/Documents/github/Bone_CLS/working/analysis/modality_overview
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Load, fix column swap, combine, dedupe  (same convention as clf analysis)
# ---------------------------------------------------------------------------

def load_and_fix(path: Path) -> pd.DataFrame:
    """Load a review CSV and swap the mislabelled Paciente / Serie columns."""
    df = pd.read_csv(path, low_memory=False)
    if not {"Paciente", "Serie"}.issubset(df.columns):
        raise ValueError(f"{path}: expected columns 'Paciente' and 'Serie'")
    df = df.rename(columns={"Paciente": "Serie", "Serie": "Paciente"})
    df["__source"] = path.name
    return df


def combine(csv_paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in csv_paths:
        df = load_and_fix(path)
        print(f"Loaded {path.name}: {len(df):,} rows")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["Nombre DICOM"], keep="first")
    after = len(combined)
    print(f"Combined: {before:,} -> {after:,} after dedupe on 'Nombre DICOM' "
          f"({before - after:,} duplicates removed)\n")
    return combined.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Modality label construction
# ---------------------------------------------------------------------------

def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def build_modality(df: pd.DataFrame, normalise: bool) -> pd.Series:
    """
    Combine the three truth columns into a single 'W-FS-C' modality label.

    When `normalise` is True:
        * FS 'Y-STIR' -> 'Y'
        * C dimension collapsed to 'N' for every weighting except T1W
          (contrast is only meaningful for T1W), so 'T2W-Y--' == 'T2W-Y-N'.
    """
    w  = _norm(df["Clase W Final"])
    fs = _norm(df["Clase FS Final"])
    c  = _norm(df["Clase C Final"])

    if normalise:
        fs = fs.replace({"Y-STIR": "Y"})
        # Contrast only applies to T1W; unify '-'/'N' -> 'N' elsewhere.
        c = c.where(w == "T1W", "N")

    return w + "-" + fs + "-" + c


# ---------------------------------------------------------------------------
# Distribution helpers
# ---------------------------------------------------------------------------

def value_counts_frame(s: pd.Series, name: str) -> pd.DataFrame:
    vc = s.value_counts(dropna=False)
    out = pd.DataFrame({name: vc.index, "count": vc.values})
    out["percent"] = (out["count"] / out["count"].sum() * 100).round(2)
    return out


def bar_plot(freq: pd.DataFrame, label_col: str, title: str, out_path: Path) -> None:
    freq = freq.sort_values("count", ascending=True)
    has_subjects = "subjects" in freq.columns
    fig, ax = plt.subplots(figsize=(9, max(3, 0.45 * len(freq) + 1)))
    ax.barh(freq[label_col].astype(str), freq["count"], color="#4C72B0")
    for y, row in enumerate(freq.itertuples(index=False)):
        cnt = getattr(row, "count")
        pct = getattr(row, "percent")
        if has_subjects:
            label = f" {cnt:,} series · {getattr(row, 'subjects'):,} subjects ({pct:.1f}%)"
        else:
            label = f" {cnt:,} ({pct:.1f}%)"
        ax.text(cnt, y, label, va="center", fontsize=8)
    ax.set_xlabel("number of series")
    ax.set_title(title)
    ax.margins(x=0.25)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  bar chart -> {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("csvs", type=Path, nargs="+",
                        help="Two or more review CSVs (Paciente/Serie swapped "
                             "in header); concatenated and de-duplicated")
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parent / "modality_overview",
                        help="Where to write CSVs and charts (default: ./modality_overview)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="white")

    df = combine(args.csvs)

    # Modality labels (equivalent modalities folded together — see module
    # docstring: Y-STIR -> Y, and contrast collapsed to N for non-T1W).
    df["modality"]     = build_modality(df, normalise=True)
    df["modality_raw"] = build_modality(df, normalise=False)

    # --- Combined modality distribution, series + subject counts ---
    freq_norm = value_counts_frame(df["modality"], "modality")
    # Distinct subjects (Paciente) having at least one series of each modality.
    subj_per_mod = df.groupby("modality")["Paciente"].nunique()
    freq_norm["subjects"] = freq_norm["modality"].map(subj_per_mod).astype(int)

    freq_raw = value_counts_frame(df["modality_raw"], "modality")

    freq_norm.to_csv(args.out_dir / "modality_distribution.csv", index=False)
    freq_raw.to_csv(args.out_dir / "modality_distribution_raw.csv", index=False)

    print(f"Modality distribution — {len(df):,} series across "
          f"{df['Paciente'].nunique():,} subjects, "
          f"{len(freq_norm)} distinct modalities:")
    print(freq_norm.to_string(index=False))
    print()

    bar_plot(freq_norm, "modality",
             "Ground-truth imaging modalities",
             args.out_dir / "modality_distribution.png")

    # --- Filtered view: drop non-target weightings ---
    W_DROP = {"Other", "DW", "Localizer", "Zip/JPG"}
    keep = ~_norm(df["Clase W Final"]).isin(W_DROP)
    df_f = df.loc[keep]

    freq_f = value_counts_frame(df_f["modality"], "modality")
    subj_f = df_f.groupby("modality")["Paciente"].nunique()
    freq_f["subjects"] = freq_f["modality"].map(subj_f).astype(int)
    freq_f.to_csv(args.out_dir / "modality_distribution_filtered.csv", index=False)

    print(f"Modality distribution (excluding {sorted(W_DROP)}) — "
          f"{len(df_f):,} series across {df_f['Paciente'].nunique():,} subjects:")
    print(freq_f.to_string(index=False))
    print()

    bar_plot(freq_f, "modality",
             "Ground-truth imaging modalities (T1W / T2W / T2* / PD)",
             args.out_dir / "modality_distribution_filtered.png")

    # --- Marginal distributions of each dimension ---
    w_freq  = value_counts_frame(_norm(df["Clase W Final"]), "weighting")
    fs_freq = value_counts_frame(_norm(df["Clase FS Final"]).replace({"Y-STIR": "Y"}),
                                 "fat_suppression")
    c_freq  = value_counts_frame(_norm(df["Clase C Final"]), "contrast")

    w_freq.to_csv(args.out_dir / "weighting_distribution.csv", index=False)
    fs_freq.to_csv(args.out_dir / "fs_distribution.csv", index=False)
    c_freq.to_csv(args.out_dir / "c_distribution.csv", index=False)

    print("Weighting (Clase W Final) distribution:")
    print(w_freq.to_string(index=False))
    print()

    bar_plot(w_freq, "weighting",
             "Ground-truth weighting (Clase W Final)",
             args.out_dir / "weighting_distribution.png")

    print(f"\nAll outputs written to {args.out_dir}")


if __name__ == "__main__":
    main()
