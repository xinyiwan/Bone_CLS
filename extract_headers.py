"""
Extract DICOM headers from a local bone MRI dataset and write to CSV.

Expected directory structure:
    DATADIR/
    └── <subject>/
        └── <session>/
            └── <scan>/
                ├── image001.dcm
                └── ...

Only the first DICOM file in each scan folder is read (all slices share the
same acquisition-level tags). One CSV row is written per scan.

Set DATADIR and OUTPUT_CSV before running, then:
    python extract_headers.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pydicom

DATADIR = Path(".")
OUTPUT_CSV = Path("dicom_headers.csv")


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _get(ds: pydicom.Dataset, tag: tuple, default=None):
    """Safely return the .value of a DICOM tag, or *default* if absent."""
    try:
        return ds[tag].value
    except KeyError:
        return default


def _to_str(value) -> str | None:
    """Convert pydicom multi-value objects (DSfloat, MultiValue…) to plain str."""
    if value is None:
        return None
    if isinstance(value, (str, int, float)):
        return value
    # pydicom sequences, MultiValue, etc.
    return str(value)


# ---------------------------------------------------------------------------
# Tag-specific processing functions (adapted from xnattools/utils_dcmsort.py)
# ---------------------------------------------------------------------------

def get_manufacturer(ds: pydicom.Dataset) -> str | None:
    value = _get(ds, (0x08, 0x70))
    if value is None:
        return None
    v = str(value).upper()
    if "SIEMENS" in v:
        return "Siemens"
    if "PHILIPS" in v:
        return "Philips"
    if "GE" in v:
        return "GE Medical"
    if "TOSHIBA" in v:
        return "Toshiba"
    return str(value) or None


def get_age(ds: pydicom.Dataset) -> str | None:
    value = _get(ds, (0x10, 0x1010))
    if value is None:
        return None
    return str(value)[:3]  # strip trailing 'Y'


def get_gender(ds: pydicom.Dataset) -> int | str | None:
    value = _get(ds, (0x10, 0x40))
    if value == "M":
        return 0
    if value == "F":
        return 1
    return value  # None or unexpected string


def get_pixel_spacing(ds: pydicom.Dataset) -> float | None:
    value = _get(ds, (0x28, 0x30))
    if value is None:
        return None
    try:
        # MultiValue → take the first element (row spacing)
        return float(value[0]) if hasattr(value, "__getitem__") else float(value)
    except (ValueError, TypeError):
        return None


def get_orientation_type(ds: pydicom.Dataset) -> int | None:
    """
    Returns:
        0 = 3D volume, 1 = axial, 2 = coronal, 3 = sagittal, 4 = oblique, 5 = 4D
    """
    try:
        acq_type = str(ds[0x18, 0x23].value)
        if acq_type == "3D":
            return 0
        if acq_type == "4D":
            return 5
    except KeyError:
        pass

    orientation = _get(ds, (0x20, 0x37))
    if orientation is None:
        return None

    x_vec = np.abs(np.array([float(v) for v in orientation[0:3]]))
    y_vec = np.abs(np.array([float(v) for v in orientation[3:6]]))
    xi, yi = int(np.argmax(x_vec)), int(np.argmax(y_vec))

    return {(0, 1): 1, (0, 2): 2, (1, 2): 3}.get((xi, yi), 4)


# ---------------------------------------------------------------------------
# Tag table: (csv_column_name, dicom_tag_or_None, process_func_or_None)
# If process_func is given it receives the full Dataset; tag is ignored.
# ---------------------------------------------------------------------------

TAGS: list[tuple[str, tuple | None, object]] = [
    # --- identification / scanner -----------------------------------------
    ("modality",            (0x08, 0x60),   None),
    ("manufacturer",        (0x08, 0x70),   get_manufacturer),
    ("station_name",        (0x08, 0x1010), None),
    ("model_name",          (0x08, 0x1090), None),
    # --- study / series ---------------------------------------------------
    ("study_date",          (0x08, 0x20),   None),
    ("series_description",  (0x08, 0x103E), None),
    ("study_description",   (0x08, 0x1030), None),
    ("protocol_name",       (0x18, 0x1030), None),
    # --- patient ----------------------------------------------------------
    ("patient_name",        (0x10, 0x10),   None),
    ("patient_ID",          (0x10, 0x20),   None),
    ("birthdate",           (0x10, 0x30),   None),
    ("age",                 (0x10, 0x1010), get_age),
    ("gender",              (0x10, 0x40),   get_gender),
    # --- acquisition parameters -------------------------------------------
    ("scanning_sequence",   (0x18, 0x20),   None),
    ("sequence_variant",    (0x18, 0x21),   None),
    ("scan_options",        (0x18, 0x22),   None),
    ("acquisition_type",    (0x18, 0x23),   None),
    ("sequence_name",       (0x18, 0x24),   None),
    ("slice_thickness",     (0x18, 0x50),   None),
    ("repetition_time_ms",  (0x18, 0x80),   None),
    ("echo_time_ms",        (0x18, 0x81),   None),
    ("inversion_time_ms",   (0x18, 0x82),   None),
    ("number_of_averages",  (0x18, 0x83),   None),
    ("magnetic_field_str",  (0x18, 0x87),   None),
    ("echo_train_length",   (0x18, 0x91),   None),
    ("flip_angle",          (0x18, 0x1314), None),
    ("coil",                (0x18, 0x1250), None),
    ("encoding_direction",  (0x18, 0x1312), None),
    ("patient_position",    (0x18, 0x5100), None),
    # --- geometry ---------------------------------------------------------
    ("rows",                (0x28, 0x10),   None),
    ("columns",             (0x28, 0x11),   None),
    ("pixel_spacing",       (0x28, 0x30),   get_pixel_spacing),
    ("spacing_btw_slices",  (0x18, 0x88),   None),
    ("orientation_type",    None,            get_orientation_type),
]

COLUMNS = ["subject", "session", "scan", "n_slices"] + [t[0] for t in TAGS]


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_scan_headers(scan_dir: Path) -> dict:
    """Read the first DICOM in *scan_dir* and return a flat dict of tag values.

    Only the first file is read; acquisition-level tags are identical across
    all slices in the same series.
    """
    dcm_files = sorted(scan_dir.glob("*.dcm"))
    if not dcm_files:
        raise FileNotFoundError(f"No .dcm files in {scan_dir}")

    ds = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)

    row: dict = {"n_slices": len(dcm_files)}
    for col_name, tag, func in TAGS:
        if func is not None:
            value = func(ds)
        else:
            value = _get(ds, tag)
        row[col_name] = _to_str(value)

    return row


# ---------------------------------------------------------------------------
# Dataset traversal
# ---------------------------------------------------------------------------

def find_scan_dirs(datadir: Path):
    """Yield ``(subject, session, scan, scan_dir)`` for every folder with .dcm files.

    The path components relative to *datadir* are interpreted as:
        3+ levels deep : parts[0]=subject, parts[1]=session, parts[2]=scan
        2 levels deep  : parts[0]=subject, parts[1]=scan (session = scan name)
        1 level deep   : subject = scan name, session = ""
    """
    scan_dirs = sorted({f.parent for f in datadir.rglob("*.dcm") if f.is_file()})
    for scan_dir in scan_dirs:
        parts = scan_dir.relative_to(datadir).parts
        if len(parts) >= 3:
            subject, session, scan = parts[0], parts[1], "/".join(parts[2:])
        elif len(parts) == 2:
            subject, session, scan = parts[0], "", parts[1]
        elif len(parts) == 1:
            subject, session, scan = parts[0], "", parts[0]
        else:
            subject, session, scan = "", "", ""
        yield subject, session, scan, scan_dir


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_all_headers(datadir: Path, output_csv: Path) -> None:
    """Walk *datadir*, extract DICOM headers for every scan, and write a CSV.

    Args:
        datadir:    Root directory that contains subject folders.
        output_csv: Destination CSV file (created or overwritten).
    """
    rows: list[dict] = []

    for subject, session, scan, scan_dir in find_scan_dirs(datadir):
        print(f"Processing  {scan_dir} …")
        try:
            row = extract_scan_headers(scan_dir)
        except Exception as e:
            print(f"  Skipped ({e})")
            continue
        row["subject"] = subject
        row["session"] = session
        row["scan"]    = scan
        rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows → {output_csv}")


if __name__ == "__main__":
    extract_all_headers(DATADIR, OUTPUT_CSV)
