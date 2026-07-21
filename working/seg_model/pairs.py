"""
Shared dataset discovery for the segmentation sub-project.

Discovery is **segmentation-driven**: the manual masks under
``<session>/segmentation_history/segs/<scan>_seg.nii.gz`` define the labelled set
(some images are excluded during segmentation), and each mask is traced back to
its image at ``<session>/<scan>/images.nii.gz``.

Also holds the plane/sequence helpers used by both the analyzer and the
nnU-Net converter, so the two stay consistent.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import nibabel as nib
except Exception as e:                                  # pragma: no cover
    raise SystemExit(f"nibabel is required: {e}")


IMAGE_NAME = "images.nii.gz"
# Mask sources, in order of preference:
#   reviewed (radiologist):  <session>/review/<xxx>/segs/<scan>_seg.nii.gz
#   history  (unreviewed) :  <session>/segmentation_history/segs/<scan>_seg.nii.gz
SEG_SUBDIR = ("segmentation_history", "segs")
REVIEW_DIR = "review"
# Most masks are *_seg.nii.gz; a few reviewed ones are *_seg.nii. Prefer .nii.gz.
SEG_SUFFIXES = ("_seg.nii.gz", "_seg.nii")


def _strip_seg_suffix(name: str):
    for suf in SEG_SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)]
    return None

# Fallback sequence parsing from the scan-folder name (used only when no
# ground-truth table is supplied). Order matters: fat-sat/contrast first.
SEQ_PATTERNS = [
    ("T1W_FS_C", re.compile(r"T1.*(FS|FAT).*(C|GD|CE|POST)|(C|GD|POST).*T1.*(FS|FAT)", re.I)),
    ("T1W_C",    re.compile(r"T1.*(C|GD|CE|POST)|(C|GD|POST).*T1", re.I)),
    ("T1W_FS",   re.compile(r"T1.*(FS|FAT|STIR)", re.I)),
    ("T2W_FS",   re.compile(r"T2.*(FS|FAT)|STIR|DPFS|PDFS|PD.*FS", re.I)),
    ("T1W",      re.compile(r"T1", re.I)),
    ("T2W",      re.compile(r"T2|PD|DP", re.I)),
    ("DWI",      re.compile(r"DWI|DIFF|ADC", re.I)),
]

PLANE_PATTERNS = [
    ("sagittal", re.compile(r"SAG", re.I)),
    ("coronal",  re.compile(r"COR", re.I)),
    ("axial",    re.compile(r"AX|TRA|TRANS", re.I)),
]


def plane_from_affine(affine: np.ndarray, zooms) -> str:
    """Infer the acquisition plane from the affine + spacing.

    The slice axis is the one with the largest spacing (slice thickness); its
    dominant anatomical direction decides the plane:
        L/R -> sagittal, A/P -> coronal, S/I -> axial.
    """
    zooms = np.asarray(zooms[:3], dtype=float)
    slice_axis = int(np.argmax(zooms))
    code = nib.aff2axcodes(affine)[slice_axis].upper()
    if code in ("L", "R"):
        return "sagittal"
    if code in ("A", "P"):
        return "coronal"
    if code in ("S", "I"):
        return "axial"
    return "unknown"


def sequence_from_name(scan_name: str) -> str:
    for label, pat in SEQ_PATTERNS:
        if pat.search(scan_name):
            return label
    return "unknown"


def plane_from_name(scan_name: str) -> str:
    for label, pat in PLANE_PATTERNS:
        if pat.search(scan_name):
            return label
    return "unknown"


def resolve_sequence(scan: str, subject: str,
                     seq_lookup: Optional[dict]) -> str:
    """Sequence label: reviewed table first (keyed by subject+scan), else filename."""
    if seq_lookup is not None:
        hit = seq_lookup.get((subject, scan))
        if hit and hit != "unknown":
            return hit
    return sequence_from_name(scan)


def load_plane_table(path: Path) -> dict:
    """Build {(subject, scan): plane} from an analyze_dataset.py per_scan.csv.

    Uses the 'plane' column, which analyze_dataset.py derives from the affine
    via plane_from_affine -- far more reliable than filename pattern matching
    (scan names don't reliably encode SAG/COR/AX).
    """
    df = pd.read_csv(path, low_memory=False)
    need = ["subject", "scan", "plane"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} missing columns {missing} (have: {list(df.columns)})")
    lookup = {}
    for _, r in df.iterrows():
        key = (str(r["subject"]).strip(), str(r["scan"]).strip())
        lookup[key] = str(r["plane"]).strip()
    return lookup


def resolve_plane(scan: str, subject: str, plane_lookup: Optional[dict],
                  affine: Optional[np.ndarray] = None, zooms=None) -> str:
    """Plane label: per_scan.csv table first (affine-derived, reliable), else
    computed directly from the image affine, else filename pattern (last resort)."""
    if plane_lookup is not None:
        hit = plane_lookup.get((subject, scan))
        if hit and hit != "unknown":
            return hit
    if affine is not None and zooms is not None:
        plane = plane_from_affine(affine, zooms)
        if plane != "unknown":
            return plane
    return plane_from_name(scan)


def resolve_seg(session_dir: Path, scan: str):
    """Best mask for a scan: radiologist-reviewed if present, else history.

    Tries both *_seg.nii.gz and *_seg.nii (prefer .gz). Returns (seg_path,
    source) with source "reviewed" or "history", or (None, None) if neither.
    """
    for suf in SEG_SUFFIXES:                              # reviewed wins over history
        rev = sorted(session_dir.glob(f"{REVIEW_DIR}/*/segs/{scan}{suf}"))
        if rev:
            return rev[0], "reviewed"
    for suf in SEG_SUFFIXES:
        hist = session_dir.joinpath(*SEG_SUBDIR, f"{scan}{suf}")
        if hist.exists():
            return hist, "history"
    return None, None


def find_pairs(root: Path):
    """Yield (subject, session, scan, image_path_or_None, seg_path, seg_source).

    Discovery is segmentation-driven over BOTH mask sources (history and
    review). For each (session, scan) seen in either source, the *reviewed* mask
    wins when available, otherwise the history mask is used. image_path is None
    if the mask's image was excluded/removed.
    """
    seen, cands = set(), []
    for seg_path in root.rglob("*_seg.nii*"):            # *_seg.nii and *_seg.nii.gz
        scan = _strip_seg_suffix(seg_path.name)
        if scan is None:
            continue
        p = seg_path.parent
        if p.name != "segs":
            continue
        if p.parent.name == SEG_SUBDIR[0]:               # segmentation_history/segs
            session_dir = p.parent.parent
        elif p.parent.parent.name == REVIEW_DIR:         # review/<xxx>/segs
            session_dir = p.parent.parent.parent
        else:
            continue
        key = (str(session_dir), scan)
        if key in seen:
            continue
        seen.add(key)
        cands.append((session_dir, scan))

    for session_dir, scan in sorted(cands, key=lambda c: (str(c[0]), c[1])):
        seg_path, source = resolve_seg(session_dir, scan)
        if seg_path is None:
            continue
        rel = session_dir.relative_to(root).parts
        subject = rel[0] if len(rel) >= 1 else session_dir.name
        session = rel[1] if len(rel) >= 2 else ""
        image_path = session_dir / scan / IMAGE_NAME
        yield (subject, session, scan,
               image_path if image_path.exists() else None, seg_path, source)


def _composite_sequence(w: str, fs: str, c: str) -> str:
    """Combine the reviewed W / FS / C finals into one sequence-type label.

        T1W -> T1W_{FS|nFS}_{CE|nCE}
        T2W -> T2W_{FS|nFS}
        DW  -> DWI
        else (Other, T2*, PD, ...) -> the raw W value
    'Y-STIR' counts as fat-sat; anything not 'Y' counts as non-FS / non-CE.
    """
    w, fs, c = (str(x).strip() for x in (w, fs, c))
    is_fs = fs.startswith("Y")          # Y or Y-STIR
    is_ce = c == "Y"
    if w == "T1W":
        return f"T1W_{'FS' if is_fs else 'nFS'}_{'CE' if is_ce else 'nCE'}"
    if w == "T2W":
        return f"T2W_{'FS' if is_fs else 'nFS'}"
    if w in ("DW", "DWI"):
        return "DWI"
    return w or "unknown"


def load_sequence_table(path: Path) -> dict:
    """Build {(subject, scan): sequence-type} from clf_perf/combined_reviewed.csv.

    That file has un-swapped columns: 'Paciente' (subject), 'Serie' (scan name),
    and 'Clase W/FS/C Final'. The sequence type is their composite.
    """
    df = pd.read_csv(path, low_memory=False)
    need = ["Paciente", "Serie", "Clase W Final", "Clase FS Final", "Clase C Final"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} missing columns {missing} (have: {list(df.columns)})")
    lookup = {}
    for _, r in df.iterrows():
        key = (str(r["Paciente"]).strip(), str(r["Serie"]).strip())
        lookup[key] = _composite_sequence(r["Clase W Final"], r["Clase FS Final"],
                                          r["Clase C Final"])
    return lookup
