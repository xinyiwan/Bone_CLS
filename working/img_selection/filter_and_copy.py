"""
Filter combined_reviewed.csv from clf_perf and copy the selected NIfTI series
(plus any pre-existing segmentation files in the same series folder) to a new
destination root.

A row is selected when its 'Clase W Final' is NOT in the EXCLUDED set:
    {"Other", "DW", "Localizer", "Zip/JPG"}

Source layout (hard-coded):
    /home/ext_xinwan/Bone_AI/tmp_data_nifti/
        ADQUISICIONES/<Paciente>/<Estudio>/<Serie>/*       (Review_Sequence_Classifier.csv)
        ADQUISICIONES_02_03_2026/<Paciente>/<Estudio>/<Serie>/*  (Review_Sequence_Classifier_n.csv)

Destination layout merges both source batches at the subject level:
    <dst>/<Paciente>/<Estudio>/<Serie>/*
(The two ADQUISICIONES batches share 3 Pacientes but no (Paciente, Estudio,
Serie) tuples — so the merge is collision-free.)

Usage:
    python filter_and_copy.py --dst /path/to/output [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

CSV_PATH = Path("/Users/xinyi/Documents/github/Bone_CLS/clf_perf/combined_reviewed.csv")
SRC_ROOT = Path("/home/ext_xinwan/Bone_AI/tmp_data_nifti")

EXCLUDED_W = {"Other", "DW", "Localizer", "Zip/JPG"}

SOURCE_TO_ADQ = {
    "Review_Sequence_Classifier.csv":   "ADQUISICIONES",
    "Review_Sequence_Classifier_n.csv": "ADQUISICIONES_02_03_2026",
}


def select_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    kept = [r for r in rows if (r.get("Clase W Final") or "").strip() not in EXCLUDED_W]
    return rows, kept


def resolve_series_dir(row: dict) -> Path | None:
    """Return the source Serie directory for a row, or None if __source is unknown."""
    adq = SOURCE_TO_ADQ.get((row.get("__source") or "").strip())
    if adq is None:
        return None
    return SRC_ROOT / adq / row["Paciente"] / row["Estudio"] / row["Serie"]


def dest_series_dir(row: dict, dst_root: Path) -> Path:
    return dst_root / row["Paciente"] / row["Estudio"] / row["Serie"]


def copy_series(src: Path, dst: Path, dry_run: bool) -> tuple[int, int]:
    """
    Copy every file under src into dst (recursive), preserving directory layout.
    Returns (files_copied, bytes_copied). Skips files that already exist with
    matching size.
    """
    if not src.exists():
        return (0, 0)

    files_copied = 0
    bytes_copied = 0
    for item in src.rglob("*"):
        if not item.is_file():
            continue
        rel = item.relative_to(src)
        target = dst / rel
        if target.exists() and target.stat().st_size == item.stat().st_size:
            continue
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
        files_copied += 1
        bytes_copied += item.stat().st_size
    return (files_copied, bytes_copied)


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dst", type=Path, required=True,
                   help="Destination root directory (will be created if missing).")
    p.add_argument("--csv", type=Path, default=CSV_PATH,
                   help=f"Path to combined_reviewed.csv (default: {CSV_PATH}).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be copied without touching the filesystem.")
    args = p.parse_args()

    if not args.csv.exists():
        print(f"ERROR: CSV not found at {args.csv}", file=sys.stderr)
        return 1

    all_rows, kept = select_rows(args.csv)
    excluded_counts = Counter(
        (r.get("Clase W Final") or "").strip()
        for r in all_rows
        if (r.get("Clase W Final") or "").strip() in EXCLUDED_W
    )
    kept_counts = Counter((r.get("Clase W Final") or "").strip() for r in kept)

    print(f"CSV rows           : {len(all_rows):,}")
    print(f"Excluded (W ∈ {sorted(EXCLUDED_W)}):")
    for k, v in sorted(excluded_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {v:5d}  {k}")
    print(f"Selected           : {len(kept):,}")
    for k, v in sorted(kept_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {v:5d}  {k}")

    # Deduplicate by series dir — a series can appear multiple times in the CSV
    # only via overlapping source rows, but we still want to copy each folder once.
    series_dirs: dict[Path, Path] = {}
    bad_source = 0
    for r in kept:
        src = resolve_series_dir(r)
        if src is None:
            bad_source += 1
            continue
        dst = dest_series_dir(r, args.dst)
        series_dirs.setdefault(src, dst)

    print(f"\nUnique series dirs : {len(series_dirs):,}")
    if bad_source:
        print(f"  (skipped {bad_source} rows with unknown __source)")

    if args.dry_run:
        print("\n--dry-run: no files will be written.")

    missing = 0
    total_files = 0
    total_bytes = 0
    for src, dst in series_dirs.items():
        if not src.exists():
            missing += 1
            continue
        n_files, n_bytes = copy_series(src, dst, args.dry_run)
        total_files += n_files
        total_bytes += n_bytes

    print(f"\nCopied series dirs : {len(series_dirs) - missing:,}")
    print(f"Missing on disk    : {missing:,}")
    print(f"Files {'would be ' if args.dry_run else ''}copied : {total_files:,}")
    print(f"Bytes {'would be ' if args.dry_run else ''}copied : {fmt_bytes(total_bytes)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
