"""
Compare predictions between:
  - DICOM classifier  (DCM clf)
  - Physics-informed method  (DICT_Physics / label_physics.py)

Comparison axes
---------------
  modality   – phys_sequence  vs  dcm_modality   (T1W | T2W | T2* | PD | DWI | …)
  fat_sat    – phys_fat_sat   vs  dcm_fat_sat    (FS or STIR → Y; ""  → N)
  contrast   – phys_contrast  vs  dcm_contrast   (Contrast   → Y; ""  → N)

phys_acquisition (FSE | SE | GRE | IR | FLAIR) is carried through as
an informational column only — it does NOT enter the comparison labels.

Usage
-----
    python compare_physics.py dcm_clf.csv physics_labels.csv [--out-dir DIR]
"""

import argparse
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_dcm_clf(path: str) -> pd.DataFrame:
    """Load DICOM classifier CSV and normalise to a common schema."""
    df = pd.read_csv(path, low_memory=False)

    df = df.rename(columns={
        "Paciente": "scan",
        "Estudio":  "session",
        "Serie":    "subject",
    })
    df = df.rename(columns={
        "Predicción Clases W":  "dcm_modality",
        "Predicción Clases FS": "dcm_fat_sat",
        "Predicción Clases C":  "dcm_contrast",
    })

    keep = ["subject", "session", "scan",
            "dcm_modality", "dcm_fat_sat", "dcm_contrast"]
    for prob_col in ["Predicción Clases W P",
                     "Predicción Clases FS P",
                     "Predicción Clases C P"]:
        if prob_col in df.columns:
            keep.append(prob_col)

    df = df[[c for c in keep if c in df.columns]].copy()
    for col in ["dcm_modality", "dcm_fat_sat", "dcm_contrast"]:
        df[col] = df[col].astype(str).str.strip()
    return df


def load_physics(path: str) -> pd.DataFrame:
    """Load physics-label CSV (output of label_physics.py) and normalise."""
    df = pd.read_csv(path, low_memory=False)

    # phys_fat_sat: "FS" or "STIR" both count as fat-suppressed
    def yn_fat(val) -> str:
        v = str(val).strip()
        return "Y" if v in ("FS", "STIR") else "N"

    # phys_contrast: "Contrast" → Y
    def yn_contrast(val) -> str:
        return "Y" if str(val).strip() == "Contrast" else "N"

    df["phys_fat_sat_yn"]  = df["phys_fat_sat"].apply(yn_fat)
    df["phys_contrast_yn"] = df["phys_contrast"].apply(yn_contrast)

    keep = ["subject", "session", "scan",
            "phys_sequence", "phys_acquisition",   # acquisition kept as info only
            "phys_fat_sat", "phys_fat_sat_yn",
            "phys_contrast", "phys_contrast_yn"]
    df = df[[c for c in keep if c in df.columns]].copy()

    df["phys_sequence"] = df["phys_sequence"].astype(str).str.strip()
    return df


# ---------------------------------------------------------------------------
# Combined labels  (modality + fat_sat + contrast)
# ---------------------------------------------------------------------------

_SUBTYPES = {
    ("T1W",  "N", "N"): "T1W-no-no",
    ("T1W",  "Y", "N"): "T1W-fs-no",
    ("T1W",  "N", "Y"): "T1W-no-c",
    ("T1W",  "Y", "Y"): "T1W-fs-c",
    ("T2W",  "N", "N"): "T2W-no-no",
    ("T2W",  "Y", "N"): "T2W-fs-no",
    ("T2W",  "N", "Y"): "T2W-no-c",
    ("T2W",  "Y", "Y"): "T2W-fs-c",
    ("T2*",  "N", "N"): "T2*-no-no",
    ("T2*",  "Y", "N"): "T2*-fs-no",
    ("T2*",  "N", "Y"): "T2*-no-c",
    ("T2*",  "Y", "Y"): "T2*-fs-c",
    ("PD",   "N", "N"): "PD-no-no",
    ("PD",   "Y", "N"): "PD-fs-no",
    ("PD",   "N", "Y"): "PD-no-c",
    ("PD",   "Y", "Y"): "PD-fs-c",
    ("DWI",  "N", "N"): "DWI-no-no",
    ("DWI",  "Y", "N"): "DWI-fs-no",
}


def _combined(modality: str, fs: str, contrast: str) -> str:
    return _SUBTYPES.get((modality, fs, contrast), f"{modality}-{fs}-{contrast}")


def make_label_dcm(row) -> str:
    fs = "Y" if row["dcm_fat_sat"] == "Y" else "N"
    c  = "Y" if row["dcm_contrast"] == "Y" else "N"
    return _combined(row["dcm_modality"], fs, c)


def make_label_physics(row) -> str:
    return _combined(row["phys_sequence"],
                     row["phys_fat_sat_yn"],
                     row["phys_contrast_yn"])


# ---------------------------------------------------------------------------
# Stats & plotting (identical structure to compare.py)
# ---------------------------------------------------------------------------

def agreement_stats(tp, tn, fp, fn) -> dict:
    total     = tp + tn + fp + fn
    agree     = tp + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else float("nan"))
    return dict(tp=tp, tn=tn, fp=fp, fn=fn,
                total=total, agree=agree,
                precision=precision, recall=recall, f1=f1)


def print_agreement(label: str, stats: dict) -> None:
    t, a = stats["total"], stats["agree"]
    pct  = a / t * 100 if t > 0 else float("nan")
    print(f"\n  [{label}]  n={t}")
    print(f"    Both positive (TP): {stats['tp']}  |  Both negative (TN): {stats['tn']}")
    print(f"    Phys=Y DCM=N  (FP): {stats['fp']}  |  Phys=N DCM=Y  (FN): {stats['fn']}")
    print(f"    Agreement : {a}/{t} ({pct:.1f}%)  "
          f"Precision={stats['precision']:.3f}  "
          f"Recall={stats['recall']:.3f}  "
          f"F1={stats['f1']:.3f}")


def plot_crosstab(cross: pd.DataFrame, out_path: Path) -> None:
    data = cross.drop(index="TOTAL", columns="TOTAL", errors="ignore")

    col_totals = data.sum(axis=0)
    annot = data.apply(
        lambda col: col.map(lambda v: f"{v}\n({v/col_totals[col.name]*100:.0f}%)")
        if col_totals[col.name] > 0 else col.map(str)
    )

    fig, ax = plt.subplots(figsize=(max(6, len(data.columns) * 1.2),
                                    max(4, len(data.index) * 0.8)))
    sns.heatmap(
        data, annot=annot, fmt="", cmap="Blues",
        linewidths=0.5, linecolor="grey",
        cbar_kws={"label": "count"},
        ax=ax,
    )
    ax.set_xlabel("DCM classifier", fontsize=11)
    ax.set_ylabel("Physics method", fontsize=11)
    ax.set_title("Prediction agreement\n(physics rows × DCM clf cols)", fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Heatmap saved to: {out_path}")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

# Subtypes included in per-subtype binary agreement.
# Covers all modality × fat_sat × contrast combinations the DCM clf can produce.
_COMPARE_SUBTYPES = [
    # T1W
    "T1W-no-no", "T1W-fs-no", "T1W-no-c", "T1W-fs-c",
    # T2W (DCM clf does not separate CE for T2W but physics can produce it)
    "T2W-no-no", "T2W-fs-no", "T2W-no-c", "T2W-fs-c",
    # T2*
    "T2*-no-no", "T2*-fs-no", "T2*-no-c", "T2*-fs-c",
    # PD
    "PD-no-no",  "PD-fs-no",
    # DWI
    "DWI-no-no", "DWI-fs-no",
]


def analyse(dcm_path: str, phys_path: str, out_dir: Path) -> None:
    dcm  = load_dcm_clf(dcm_path)
    phys = load_physics(phys_path)

    print(f"DCM clf rows     : {len(dcm):,}")
    print(f"Physics rows     : {len(phys):,}")

    merged = dcm.merge(phys, on=["subject", "session", "scan"], how="inner")
    print(f"\nMatched rows (inner join): {len(merged):,}")

    if merged.empty:
        print("No matching rows — check that subject/session/scan keys align.")
        return

    # ---- Combined labels --------------------------------------------------
    merged["dcm_label"]  = merged.apply(make_label_dcm,     axis=1)
    merged["phys_label"] = merged.apply(make_label_physics, axis=1)

    # ---- Crosstab  (all rows, including Localizer and Unknown) --------------
    cross = pd.crosstab(
        merged["phys_label"],
        merged["dcm_label"],
        margins=True, margins_name="TOTAL"
    )
    print(f"\n--- Combined-label crosstab  (physics rows vs DCM clf cols, "
          f"n={len(merged):,}) ---")
    print(cross.to_string())

    plot_crosstab(cross, out_dir / "crosstab_physics_vs_dcm.png")

    # ---- Per-subtype binary agreement -------------------------------------
    print("\n--- Per-subtype agreement (binary: is this subtype vs not) ---")
    all_disagree = []

    present_subtypes = [s for s in _COMPARE_SUBTYPES
                        if (merged["phys_label"] == s).any()
                        or (merged["dcm_label"]  == s).any()]

    for subtype in present_subtypes:
        dcm_pos  = merged["dcm_label"]  == subtype
        phys_pos = merged["phys_label"] == subtype

        tp = ( phys_pos &  dcm_pos).sum()
        tn = (~phys_pos & ~dcm_pos).sum()
        fp = ( phys_pos & ~dcm_pos).sum()   # physics says yes, DCM says no
        fn = (~phys_pos &  dcm_pos).sum()   # DCM says yes, physics says no

        print_agreement(subtype, agreement_stats(tp, tn, fp, fn))

        dis = merged[(phys_pos != dcm_pos)].copy()
        dis.insert(0, "subtype", subtype)
        all_disagree.append(dis)

    # ---- Save disagreements -----------------------------------------------
    if all_disagree:
        disagree_df = pd.concat(all_disagree, ignore_index=True)
        cols = ["subtype", "subject", "session", "scan",
                "phys_label", "dcm_label",
                "phys_sequence", "phys_acquisition",   # acquisition as info
                "phys_fat_sat", "phys_contrast",
                "dcm_modality", "dcm_fat_sat", "dcm_contrast"]
        out_csv = out_dir / "physics_vs_dcm_disagreements.csv"
        disagree_df[[c for c in cols if c in disagree_df.columns]].to_csv(
            out_csv, index=False
        )
        print(f"\nDisagreements saved to: {out_csv}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare physics-informed labels vs DCM classifier"
    )
    parser.add_argument("dcm_csv",   help="Path to DICOM classifier CSV")
    parser.add_argument("phys_csv",  help="Path to physics-label CSV (label_physics.py output)")
    parser.add_argument("--out-dir", default=None,
                        help="Directory for output files (default: same as dcm_csv)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.dcm_csv).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    analyse(args.dcm_csv, args.phys_csv, out_dir)
