"""
Physics-informed MRI sequence classifier.

Applies physics-based rules using DICOM acquisition parameters
(TR, TE, TI, ETL, FA, field strength, diffusion tags) to classify
each scan.  Four output columns are produced:

  phys_sequence   – T1W | T2@ | T2* | PD | DWI | STIR | FLAIR | T1W_IR |
                    GENERIC_IR | Localizer | GENERIC_GRE | UNKNOWN
  phys_acquisition – FSE | SE | GRE | ""   (empty for IR / Localizer)
  phys_fat_sat    – FS | STIR | STIR+FS | ""
  phys_contrast   – Contrast | ""

Classification steps (mutually exclusive, applied in order)
------------------------------------------------------------
1. DWI    – diffusion_b_value present, or diffusion_gradient_orientation present
2. IR     – InversionTime > 0  →  STIR / FLAIR / T1W_IR / GENERIC_IR
3. GRE    – scanning_sequence contains "GR"
4. (F)SE  – all remaining

Usage
-----
    python label_physics.py input.csv [output.csv]
"""

from __future__ import annotations

import re
import sys
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _safe_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_present(value) -> bool:
    """True when value is non-null and non-empty."""
    if value is None:
        return False
    return str(value).strip().lower() not in ("", "nan", "none", "null")


# Token-based fat-sat keywords (label_csv.py style).
# STIR included: _fat_sat_label() always prioritises STIR when TI confirms it,
# so including it here is safe and catches series named "STIR" without TI data.
# FLAIR excluded: suppresses fluid, not fat.
_FAT_SAT_TOKENS = {"FS", "FATSAT", "FATSUPP", "SPIR", "SPAIR", "FAT", "CHEMSAT", "STIR"}

# Regex fallback for forms that _tokens() would fragment (e.g. "FAT_SAT" → ["FAT","SAT"])
_FAT_SAT_RE = re.compile(r"\bFAT[_\-]SAT\b", re.IGNORECASE)


def _has_fat_sat(scan_options: str, series_desc: str = "") -> bool:
    """Fat suppression detected from series description or scan options.

    Combines token matching (label_csv.py) with a regex fallback for
    underscore/hyphen forms. Checks both series_desc and scan_options.
    """
    for text in (series_desc, scan_options):
        if set(_tokens(text)) & _FAT_SAT_TOKENS:
            return True
        if _FAT_SAT_RE.search(text):
            return True
    return False


_CONTRAST_AGENT_TOKENS = {
    "YES", "Y", "GD", "GAD", "CONTRASTE", "DOTAREM", "GADO", "MH", "MULTIHANCE",
}


def _tokens(text: str) -> list[str]:
    """Alpha-only tokens from an upper-cased string (mirrors label_csv.py)."""
    if not text:
        return []
    return [t for t in re.split(r"[^A-Z]+", text.upper()) if t]


def _has_contrast(series_desc: str, contrast_agent: str, total_dose: str) -> bool:
    """True when contrast is detected from series description or agent/dose fields."""
    upper = series_desc.upper()

    # Series description keywords: GD/GAD tokens or +C / CTE patterns
    desc_tokens = set(_tokens(series_desc))
    if {"GD", "GAD"} & desc_tokens:
        return True
    if re.search(r"\+C(?:TE)?|\bCTE\b", upper):
        return True

    # Agent field populated with a meaningful value → contrast given
    # Dose is not required: it is frequently missing even when contrast was administered.
    ca = str(contrast_agent).strip()
    dose = str(total_dose).strip()
    if ca and ca.upper() not in ("", "NONE", "NO", "0") and dose != "0":
        return True

    # Agent token matches known contrast-agent names (dose must not be explicitly 0)
    agent_tokens = set(_tokens(contrast_agent))
    if agent_tokens & _CONTRAST_AGENT_TOKENS and dose != "0":
        return True

    return False


def _acq(etl: float | None) -> str:
    """'FSE' when ETL ≥ 2, 'SE' otherwise."""
    return "FSE" if (etl is not None and etl >= 2) else "SE"


def _fat_sat_label(is_stir: bool, has_fs: bool) -> str:
    """Combine STIR-based and chemical fat-suppression into one field.

    STIR takes priority: when TI places the sequence in the STIR range,
    that is the operative fat-suppression mechanism regardless of any
    additional FS scan option.
    """
    if is_stir:
        return "STIR"
    if has_fs:
        return "FS"
    return ""


# ---------------------------------------------------------------------------
# Result builder
# ---------------------------------------------------------------------------

def _result(
    sequence: str,
    acquisition: str = "",
    fat_sat: str = "",
    contrast: str = "",
) -> dict:
    return {
        "phys_sequence":    sequence,
        "phys_acquisition": acquisition,
        "phys_fat_sat":     fat_sat,
        "phys_contrast":    contrast,
    }


# ---------------------------------------------------------------------------
# Branch classifiers — each returns a result dict
# ---------------------------------------------------------------------------

def _classify_ir(
    TI: float,
    TE: float | None,
    TR: float | None,
    B0: float | None,
    has_fs: bool,
    has_contrast: bool,
) -> dict:
    """Step 2: Inversion Recovery branch."""
    is_15T = B0 is not None and B0 < 2.0
    is_3T  = B0 is not None and B0 >= 2.0
    contrast = "Contrast" if has_contrast else ""

    # STIR — fat suppression is STIR-based; chemical FS may co-occur
    # TODO: change the range here to include more errors on STIR
    if (is_15T and 110 <= TI <= 190) or (is_3T and 150 <= TI <= 240):
        # return _result("STIR", acquisition="IR", fat_sat=_fat_sat_label(True, has_fs), contrast=contrast)
        # TODO: check if I could make STIR automatically T2W
        return _result("T2W", acquisition="IR", fat_sat=_fat_sat_label(True, has_fs), contrast=contrast)
        

    # FLAIR — fluid suppression, not fat; record chemical FS independently
    if (is_15T and 1900 <= TI <= 2600) or (is_3T and 2400 <= TI <= 3200):
        # TODO: check if I could make FLAIR automatically T2W.
        return _result("T2W", acquisition="FLAIR", fat_sat=_fat_sat_label(False, has_fs), contrast=contrast)

    # T1W_IR
    if TE is not None and TR is not None and TE <= 30 and TR < 4000:
        return _result("T1W", acquisition="IR", fat_sat=_fat_sat_label(False, has_fs), contrast=contrast)

    return _result("Unknown", acquisition="IR", fat_sat=_fat_sat_label(False, has_fs), contrast=contrast)


def _classify_gre(
    TR: float | None,
    TE: float | None,
    FA: float | None,
    B0: float | None,
    has_fs: bool,
    has_contrast: bool,
) -> dict:
    """Step 3: GRE branch."""
    fs      = "FS" if has_fs else ""
    contrast = "Contrast" if has_contrast else ""

    if TR is None or TE is None:
        return _result("Unknown_GRE", acquisition="GRE", fat_sat=fs, contrast=contrast)

    is_15T = B0 is not None and B0 < 2.0
    is_3T  = B0 is not None and B0 >= 2.0

    # Localizer: very short TR/TE with typical scout flip angle
    if TR <= 8 and TE <= 4 and FA is not None and 35 <= FA <= 100:
        return _result("Localizer", acquisition="GRE")

    # T1_GRE
    if TR <= 20 and TE <= 6:
        return _result("T1W", acquisition="GRE", fat_sat=fs, contrast=contrast)

    # T2*_GRE
    # Simplify the rules to just TE thresholds for both 1.5T and 3T.
    # changed the threshold to 6 based on inspected errors
    if TE >= 6:
        return _result("T2*", acquisition="GRE", fat_sat=fs, contrast=contrast)

    return _result("Unknown", acquisition="GRE", fat_sat=fs, contrast=contrast)


def _classify_fse(
    TR: float | None,
    TE: float | None,
    ETL: float | None,
    B0: float | None,
    has_fs: bool,
    has_contrast: bool,
) -> dict:
    """Step 4: (F)SE branch. Always returns a result."""
    acquisition = _acq(ETL)
    is_15T = B0 is not None and B0 < 2.0
    is_3T  = B0 is not None and B0 >= 2.0
    fs       = "FS" if has_fs else ""
    contrast = "Contrast" if has_contrast else ""

    seq = "Unknown"

    if TR is not None and TE is not None:
        # TE is the primary determinant; TR breaks ties.
        #
        #  TE ≤ 25 ms  (short TE — T2* suppressed)
        #    TR ≤ 1100 → T1W   (short TR, short TE)
        #    TR > 1100 → PD    (long TR, short TE — T1 suppressed too)
        #
        #  TE ≥ 70 ms  (long TE — T2 decay dominant)
        #    → T2W
        #
        #  25 < TE < 70 ms  (borderline — neither fully T2W nor fully T1W)
        #    TR < 1500 → T1W  (short TR keeps some T1 weighting)
        #    TR ≥ 1500 → T2W  (long TR, moderate TE → T2-like)

        if TE <= 25:
            seq = "T1W" if TR <= 1100 else "PD"
        elif TE >= 70:
            seq = "T2W"
        else:  # 25 < TE < 70
            seq = "T1W" if TR < 1500 else "T2W"

    return _result(seq, acquisition=acquisition, fat_sat=fs, contrast=contrast)


# ---------------------------------------------------------------------------
# Top-level row classifier
# ---------------------------------------------------------------------------

def classify_physics(row: pd.Series) -> dict:
    """Return physics-based label columns for one CSV row."""
    TR  = _safe_float(row.get("repetition_time_ms"))
    TE  = _safe_float(row.get("echo_time_ms"))
    TI  = _safe_float(row.get("inversion_time_ms"))
    ETL = _safe_float(row.get("echo_train_length"))
    FA  = _safe_float(row.get("flip_angle"))
    B0  = _safe_float(row.get("magnetic_field_str"))

    scan_options   = str(row.get("scan_options",       "") or "")
    series_desc    = str(row.get("series_description", "") or "")
    scanning_seq   = str(row.get("scanning_sequence",  "") or "").upper()
    contrast_agent = str(row.get("Contrast_Agent",     "") or "")
    total_dose     = str(row.get("Total_Dose",         "") or "")
    image_type     = str(row.get("image_type",         "") or "").upper()

    b_value     = row.get("diffusion_b_value")
    diff_orient = row.get("diffusion_gradient_orientation")

    has_fs  = _has_fat_sat(scan_options, series_desc)
    has_contrast = _has_contrast(series_desc, contrast_agent, total_dose)

    # ------------------------------------------------------------------
    # Step 0: Localizer — checked first, before any physics rules
    # Uses the same keyword logic as label_csv.py.
    # ------------------------------------------------------------------
    if re.search(r"CALI|LOC|LOCAL|SCOUT|SURVEY|CALIBRATION|ASSET", series_desc.upper()):
        return _result("Localizer")

    # ------------------------------------------------------------------
    # Step 1: DWI — if DWI stop here, do not continue to step 2
    # b_value == 0 is the non-diffusion-weighted reference volume; not DWI.
    # ------------------------------------------------------------------
    b_val_float = _safe_float(b_value)
    b_value_is_dwi = _is_present(b_value) and b_val_float is not None and b_val_float >= 0

    is_dwi = (
        b_value_is_dwi
        or "DIFFUSION" in image_type
    )
    if is_dwi:
        fs       = "FS" if has_fs else ""
        contrast = "Contrast" if has_contrast else ""
        return _result("DWI", acquisition=_acq(ETL), fat_sat=fs, contrast=contrast)

    # ------------------------------------------------------------------
    # Step 2: Inversion Recovery — if TI > 0 stop here
    # ------------------------------------------------------------------
    if TI is not None and TI > 0:
        return _classify_ir(TI, TE, TR, B0, has_fs, has_contrast)

    # ------------------------------------------------------------------
    # Step 3: GRE — if scanning_sequence contains GR stop here
    # ------------------------------------------------------------------
    if "GR" in scanning_seq:
        return _classify_gre(TR, TE, FA, B0, has_fs, has_contrast)

    # ------------------------------------------------------------------
    # Step 4: (F)SE — all remaining
    # ------------------------------------------------------------------
    return _classify_fse(TR, TE, ETL, B0, has_fs, has_contrast)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

REQUIRED_COLS = ["repetition_time_ms", "echo_time_ms", "magnetic_field_str"]
OUT_COLS = ["phys_sequence", "phys_acquisition", "phys_fat_sat", "phys_contrast"]


def label(input_csv: Path, output_csv: Path) -> None:
    df = pd.read_csv(input_csv, dtype=str).fillna("")

    before = len(df)
    df = df[df["modality"] == "MR"]
    print(f"Dropped {before - len(df)} non-MR rows ({len(df)} remaining)")

    before = len(df)
    df = df[df[REQUIRED_COLS].apply(lambda r: r.str.strip().ne("")).all(axis=1)]
    print(f"Dropped {before - len(df)} rows with missing required fields ({len(df)} remaining)")

    results = df.apply(classify_physics, axis=1, result_type="expand")
    df[OUT_COLS] = results[OUT_COLS]

    df.to_csv(output_csv, index=False)
    print(f"Labelled {len(df)} rows → {output_csv}")

    for col in OUT_COLS:
        print(f"\n{col} counts:")
        print(df[col].replace("", "(empty)").value_counts().to_string())


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        sys.exit("Usage: python label_physics.py input.csv [output.csv]")
    input_csv  = Path(args[0])
    output_csv = Path(args[1]) if len(args) > 1 else input_csv
    label(input_csv, output_csv)
