"""
Compare T1W native predictions between:
  - DICOM classifier (DCM clf)
  - Dictionary matching method (dict method)

Focuses on T1W native sequences: T1W modality, no contrast, no fat saturation.
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
        "Paciente": "subject",
        "Estudio":  "session",
        "Serie":    "scan",
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


def is_t1w_native_dcm(df: pd.DataFrame) -> pd.Series:
    """True for rows the DCM classifier calls T1W native (no C, no FS)."""
    return (
        (df["dcm_modality"] == "T1W") &
        (df["dcm_contrast"].isin(["N", "-"])) &
        (df["dcm_fat_sat"].isin(["N", "-"]))
    )


def is_t1w_native_dict(df: pd.DataFrame) -> pd.Series:
    """True for rows the dict method calls T1W native (no C, no FS)."""
    return (
        (df["dict_modality"] == "T1W") &
        (df["dict_contrast"] == "N") &
        (df["dict_fat_sat"] == "N")
    )


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyse(dcm_path: str, dict_path: str) -> None:
    dcm  = load_dcm_clf(dcm_path)
    dct  = load_dict_method(dict_path)

    print(f"DCM clf rows : {len(dcm):,}")
    print(f"Dict method rows: {len(dct):,}")

    # ---- Merge on subject / session / scan --------------------------------
    merged = dcm.merge(dct, on=["subject", "session", "scan"], how="inner")
    print(f"\nMatched rows (inner join): {len(merged):,}")

    if merged.empty:
        print("No matching rows found – check that subject/session/scan keys align.")
        return

    # ---- Boolean T1W-native flags ----------------------------------------
    merged["dcm_t1w_native"]  = is_t1w_native_dcm(merged)
    merged["dict_t1w_native"] = is_t1w_native_dict(merged)

    n_dcm_pos  = merged["dcm_t1w_native"].sum()
    n_dict_pos = merged["dict_t1w_native"].sum()
    print(f"\nT1W native positives — DCM clf: {n_dcm_pos}  |  Dict method: {n_dict_pos}")

    # ---- Confusion matrix (dict vs dcm) ----------------------------------
    tp = (merged["dcm_t1w_native"]  & merged["dict_t1w_native"]).sum()
    tn = (~merged["dcm_t1w_native"] & ~merged["dict_t1w_native"]).sum()
    fp = (~merged["dcm_t1w_native"] &  merged["dict_t1w_native"]).sum()
    fn = (merged["dcm_t1w_native"]  & ~merged["dict_t1w_native"]).sum()

    total = len(merged)
    agree = tp + tn

    print("\n--- Agreement (T1W native) ---")
    print(f"  Both agree T1W native      (TP): {tp}")
    print(f"  Both agree NOT T1W native  (TN): {tn}")
    print(f"  Dict=Y, DCM=N              (FP): {fp}")
    print(f"  Dict=N, DCM=Y              (FN): {fn}")
    print(f"  Overall agreement: {agree}/{total} ({agree/total*100:.1f}%)")

    if tp + fp + fn > 0:
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else float("nan"))
        print(f"  Precision (DCM as ref): {precision:.3f}")
        print(f"  Recall    (DCM as ref): {recall:.3f}")
        print(f"  F1 score              : {f1:.3f}")

    # ---- Breakdown: modality agreement (for ALL rows) --------------------
    print("\n--- Modality agreement (all rows) ---")
    mod_agree = (merged["dcm_modality"] == merged["dict_modality"]).sum()
    print(f"  Exact modality match: {mod_agree}/{total} ({mod_agree/total*100:.1f}%)")

    cross = pd.crosstab(
        merged["dict_modality"], merged["dcm_modality"],
        margins=True, margins_name="TOTAL"
    )
    print("\n  Crosstab  dict (rows) vs DCM clf (cols):")
    print(cross.to_string())

    # ---- Focus: rows where either method says T1W native -----------------
    either = merged["dcm_t1w_native"] | merged["dict_t1w_native"]
    t1w_df = merged[either].copy()

    if not t1w_df.empty:
        print(f"\n--- Rows flagged T1W native by at least one method ({len(t1w_df)}) ---")
        display_cols = ["subject", "session", "scan",
                        "dcm_modality", "dcm_fat_sat", "dcm_contrast",
                        "dict_modality", "dict_fat_sat", "dict_contrast",
                        "dcm_t1w_native", "dict_t1w_native"]
        print(t1w_df[[c for c in display_cols if c in t1w_df.columns]].to_string(index=False))

    # ---- Disagreement details --------------------------------------------
    disagree_df = merged[merged["dcm_t1w_native"] != merged["dict_t1w_native"]].copy()
    if not disagree_df.empty:
        out_path = Path(dcm_path).parent / "t1w_native_disagreements.csv"
        disagree_df.to_csv(out_path, index=False)
        print(f"\nDisagreements saved to: {out_path}")


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
