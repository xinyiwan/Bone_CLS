"""
Compare T1W predictions between:
  - DICOM classifier (DCM clf)
  - Dictionary matching method (dict method)

Breaks T1W into three subtypes:
  T1W-no-no  : T1W, no fat-sat, no contrast  (native)
  T1W-fs-no  : T1W, fat-sat,    no contrast
  T1W-fs-c   : T1W, fat-sat,    with contrast
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dcm_clf(path: str) -> pd.DataFrame:
    """Load DICOM classifier CSV and normalise to a common schema."""
    df = pd.read_csv(path, low_memory=False)

    # Rename join keys to canonical names
    df = df.rename(columns={
        "Paciente": "scan",
        "Estudio":  "session",
        "Serie":    "subject",
    })

    # Relevant prediction columns
    # "Predicción Clases W"  -> modality  (T1W | T2W | Other)
    # "Predicción Clases FS" -> fat_sat   (Y | N | -)
    # "Predicción Clases C"  -> contrast  (Y | N | -)
    df = df.rename(columns={
        "Predicción Clases W":  "dcm_modality",
        "Predicción Clases FS": "dcm_fat_sat",
        "Predicción Clases C":  "dcm_contrast",
    })

    # Keep only the columns we need (plus join keys)
    keep = ["subject", "session", "scan",
            "dcm_modality", "dcm_fat_sat", "dcm_contrast"]
    # Add probability columns if present
    for prob_col in ["Predicción Clases W P",
                     "Predicción Clases FS P",
                     "Predicción Clases C P"]:
        if prob_col in df.columns:
            keep.append(prob_col)
    df = df[[c for c in keep if c in df.columns]].copy()

    # Normalise: strip whitespace, upper-case flags
    for col in ["dcm_modality", "dcm_fat_sat", "dcm_contrast"]:
        df[col] = df[col].astype(str).str.strip()

    return df


def load_dict_method(path: str) -> pd.DataFrame:
    """Load dictionary-matching CSV and normalise to a common schema."""
    df = pd.read_csv(path, low_memory=False)

    # Relevant columns
    # sequence_type -> modality  (T1W | T2W | T2* | PD | localizer | …)
    # fat_sat       -> fat saturation (non-empty = Y)
    # contrast      -> contrast agent  (non-empty = Y)
    df = df.rename(columns={
        "sequence_type": "dict_modality",
        "fat_sat":       "dict_fat_sat_raw",
        "contrast":      "dict_contrast_raw",
    })

    # Normalise fat_sat and contrast to Y / N
    def yn(series: pd.Series) -> pd.Series:
        """Convert truthy/non-empty values to Y, blank/NaN to N."""
        return series.apply(
            lambda x: "N" if (pd.isna(x) or str(x).strip() in ("", "nan"))
                      else "Y"
        )

    df["dict_fat_sat"] = yn(df["dict_fat_sat_raw"])
    df["dict_contrast"] = yn(df["dict_contrast_raw"])

    keep = ["subject", "session", "scan",
            "dict_modality", "dict_fat_sat", "dict_contrast"]
    df = df[[c for c in keep if c in df.columns]].copy()

    for col in ["dict_modality", "dict_fat_sat", "dict_contrast"]:
        df[col] = df[col].astype(str).str.strip()

    return df


# T1W subtypes we care about: (fat_sat_flag, contrast_flag) -> label
T1W_SUBTYPES = {
    ("N", "N"): "T1W-no-no",
    ("Y", "N"): "T1W-fs-no",
    ("Y", "Y"): "T1W-fs-c",
    ("N", "Y"): "T1W-no-c"
}
T2W_SUBTYPES = {
    ("N", "N"): "T2W-no-no",
    ("Y", "N"): "T2W-fs-no",
    ("Y", "Y"): "T2W-fs-c",
    ("N", "Y"): "T2W-no-c"
}
# DCM clf uses "-" for "not applicable" (when modality != T1W); treat as N
DCM_NEG = {"N", "-"}


def make_combined_label_dcm(row) -> str:
    """Composite label for DCM clf row: T1W-{fs}-{c} or the modality."""
    fs = "Y" if row["dcm_fat_sat"] == "Y" else "N"
    c  = "Y" if row["dcm_contrast"] == "Y" else "N"
    if row["dcm_modality"] == "T1W":
        return T1W_SUBTYPES.get((fs, c), f"T1W-{fs}-{c}")
    elif row["dcm_modality"] == "T2W":
        return T2W_SUBTYPES.get((fs, c), f"T2W-{fs}-{c}")
    else: 
        return row["dcm_modality"]


def make_combined_label_dict(row) -> str:
    """Composite label for dict method row: T1W-{fs}-{c} or the modality."""
    fs = row["dict_fat_sat"]   # already Y/N
    c  = row["dict_contrast"]  # already Y/N
    if row["dict_modality"] == "T1W":
        return T1W_SUBTYPES.get((fs, c), f"T1W-{fs}-{c}")
    elif row["dict_modality"] == "T2W":
        return T2W_SUBTYPES.get((fs, c), f"T2W-{fs}-{c}")
    else:
        return row["dict_modality"]
    


def agreement_stats(tp, tn, fp, fn) -> dict:
    total = tp + tn + fp + fn
    agree = tp + tn
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
    print(f"    Dict=Y DCM=N  (FP): {stats['fp']}  |  Dict=N DCM=Y  (FN): {stats['fn']}")
    print(f"    Agreement : {a}/{t} ({pct:.1f}%)  "
          f"Precision={stats['precision']:.3f}  "
          f"Recall={stats['recall']:.3f}  "
          f"F1={stats['f1']:.3f}")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyse(dcm_path: str, dict_path: str) -> None:
    dcm  = load_dcm_clf(dcm_path)
    dct  = load_dict_method(dict_path)

    print(f"DCM clf rows    : {len(dcm):,}")
    print(f"Dict method rows: {len(dct):,}")

    # ---- Merge on subject / session / scan --------------------------------
    merged = dcm.merge(dct, on=["subject", "session", "scan"], how="inner")
    print(f"\nMatched rows (inner join): {len(merged):,}")

    if merged.empty:
        print("No matching rows found – check that subject/session/scan keys align.")
        return

    # ---- Composite labels ------------------------------------------------
    merged["dcm_label"]  = merged.apply(make_combined_label_dcm,  axis=1)
    merged["dict_label"] = merged.apply(make_combined_label_dict, axis=1)

    # ---- Crosstab (exclude PD rows from dict side) -----------------------
    exclude_modalities = {""}   # DCM clf never predicts these
    mask_cross = ~merged["dict_modality"].isin(exclude_modalities)

    cross = pd.crosstab(
        merged.loc[mask_cross, "dict_label"],
        merged.loc[mask_cross, "dcm_label"],
        margins=True, margins_name="TOTAL"
    )
    total = mask_cross.sum()
    print(f"\n--- Combined-label crosstab  (dict rows vs DCM clf cols, "
          f"n={total}) ---")
    print(cross.to_string())

    # ---- Per-subtype binary agreement ------------------------------------
    print("\n--- Per-subtype agreement (binary: is this subtype vs not) ---")
    all_disagree = []

    for subtype in ["T1W-no-no", "T1W-fs-no", "T1W-fs-c"]:
        dcm_pos  = merged["dcm_label"]  == subtype
        dict_pos = merged["dict_label"] == subtype

        tp = (dcm_pos  & dict_pos).sum()
        tn = (~dcm_pos & ~dict_pos).sum()
        fp = (~dcm_pos &  dict_pos).sum()   # dict says yes, DCM says no
        fn = (dcm_pos  & ~dict_pos).sum()   # DCM says yes, dict says no

        print_agreement(subtype, agreement_stats(tp, tn, fp, fn))

        # collect disagreements for this subtype
        dis = merged[(dcm_pos != dict_pos)].copy()
        dis.insert(0, "subtype", subtype)
        all_disagree.append(dis)

    # ---- Save all disagreements ------------------------------------------
    disagree_df = pd.concat(all_disagree, ignore_index=True)
    if not disagree_df.empty:
        out_path = Path(dcm_path).parent / "t1w_subtype_disagreements.csv"
        cols = ["subtype", "subject", "session", "scan",
                "dcm_label", "dict_label",
                "dcm_modality", "dcm_fat_sat", "dcm_contrast",
                "dict_modality", "dict_fat_sat", "dict_contrast"]
        disagree_df[[c for c in cols if c in disagree_df.columns]].to_csv(
            out_path, index=False
        )
        print(f"\nAll disagreements saved to: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare T1W-native predictions: DCM classifier vs dict method"
    )
    parser.add_argument("dcm_csv",  help="Path to DICOM classifier CSV")
    parser.add_argument("dict_csv", help="Path to dictionary-matching CSV")
    args = parser.parse_args()

    analyse(args.dcm_csv, args.dict_csv)
