"""
Add sequence labels to the DICOM header CSV produced by extract_headers.py.

Three new columns are appended:
  sequence_type  – T1 | T2 | T2* | DWI | PDW | PD | perfusion | localizer  (from series_description)
  fat_sat        – fatsat  (from series_description or scan_options)
  contrast       – contrast  (from series_description)

Non-matching entries are left empty.

Usage:
    python label_csv.py dicom_headers.csv          # overwrites in place
    python label_csv.py dicom_headers.csv out.csv  # write to new file
"""

import re
import sys
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokens(text: str) -> list[str]:
    """Split an upper-cased DICOM string into alpha-only tokens (digits stripped).

    e.g. "T2_FS SAG" → ["T", "FS", "SAG"]
         "T1_FSE"    → ["T", "FSE"]
    Splitting on non-letter chars ensures 'FS' != 'FSE'.
    """
    if not text:
        return []
    return [t for t in re.split(r"[^A-Z]+", text.upper()) if t]


def label_sequence_type(series_desc: str) -> str:
    # Tokens are alpha-only, so T1/T2 become "T" — search the original string
    # for those. Purely alpha markers (LOC, DP, PD) still use tokens.
    upper = series_desc.upper()

    tokens = _tokens(series_desc)
    if re.search(r"CAL|LOC|LOCAL|SCOUT|SURVEY|CALIBRATION", upper):
        return "localizer"

    # DWI / diffusion
    if re.search(r"DWI|DIFF|DIFUSION|ADC|DIFU", upper):
        return "DWI"

    # Perfusion
    if re.search(r"PERFUSION", upper):
        return "perfusion"

    # STIR is T2-weighted
    if "STIR" in tokens:
        return "T2W"

    # Use the raw string so "AX GRE T2 (MERGE)" → T2, not lost as "T"
    if re.search(r"T1", upper):
        return "T1W"
    # T2* must be checked before plain T2
    if re.search(r"T2\*", series_desc, re.IGNORECASE):
        return "T2*"
    if re.search(r"T2", upper) or re.search(r"STIR", upper):
        return "T2W"

    if "PDW" in tokens:
        return "PD"

    if any(t in ("DP", "PD") for t in tokens):
        return "PD"

    return ""


def label_fat_sat(series_desc: str, scan_options: str) -> str:
    upper = series_desc.upper()
    for text in (series_desc, scan_options):
        tokens = _tokens(text)
        if any(t in ("STIR", "FS", "FATSAT", "FATSUPP", "SPIR", "SPAIR", "FLAIR", 'FAT') for t in tokens) or re.search(r"STIR", upper):
            return "fatsat"
    return ""


def label_contrast(series_desc: str, Contrast_Agent: str, volume: str, total_dose: str) -> str:
    upper = series_desc.upper()
    tokens = _tokens(series_desc)
    agent_tokens = _tokens(Contrast_Agent.upper())
    if any(t in ("GD", "GAD") for t in tokens) or re.search(r"/+C | /+CTE", upper):
        return "contrast"
    elif any(t in ("YES", "Y", "GD", "CONTRASTE", "DOTAREM", "GADO", "GAD", "MH", "MULTIHANCE") for t in agent_tokens) \
        and total_dose != "0":
        return "contrast"
    return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

REQUIRED_COLS = [
    "repetition_time_ms",
    "echo_time_ms",
    "magnetic_field_str",
]


def label(input_csv: Path, output_csv: Path) -> None:
    df = pd.read_csv(input_csv, dtype=str).fillna("")

    before = len(df)

    # Keep only MR modality
    df = df[df["modality"] == "MR"]
    print(f"Dropped {before - len(df)} non-MR rows ({len(df)} remaining)")

    # Drop rows with any required parameter empty
    before = len(df)
    df = df[df[REQUIRED_COLS].apply(lambda r: r.str.strip().ne("")).all(axis=1)]
    print(f"Dropped {before - len(df)} rows with missing required fields ({len(df)} remaining)")

    df["sequence_type"] = df["series_description"].apply(label_sequence_type)
    df["fat_sat"]       = df.apply(
        lambda r: label_fat_sat(r["series_description"], r["scan_options"]), axis=1
    )
    df["contrast"]      = df.apply(
        lambda r: label_contrast(r["series_description"], r["Contrast_Agent"], r["Volume"], r["Total_Dose"]), axis=1
    )

    df.to_csv(output_csv, index=False)
    print(f"Labelled {len(df)} rows → {output_csv}")

    # Quick summary
    print("\nsequence_type counts:")
    print(df["sequence_type"].replace("", "(empty)").value_counts().to_string())
    print("\nfat_sat counts:")
    print(df["fat_sat"].replace("", "(empty)").value_counts().to_string())
    print("\ncontrast counts:")
    print(df["contrast"].replace("", "(empty)").value_counts().to_string())


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        sys.exit("Usage: python label_csv.py input.csv [output.csv]")
    input_csv  = Path(args[0])
    output_csv = Path(args[1]) if len(args) > 1 else input_csv
    label(input_csv, output_csv)
