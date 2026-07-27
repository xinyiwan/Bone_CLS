"""
Filter combined_reviewed.csv from clf_perf and copy the selected NIfTI series
(plus any pre-existing segmentation files in the same series folder) to a new
destination root.

A row is selected when its 'Clase W Final' is NOT in the EXCLUDED set:
    {"Other", "DW", "Localizer", "Zip/JPG"}

Source layout:
    <src-root>/<ADQUISICIONES batch>/<Paciente>/<Estudio>/<Serie>/*
Each row's '__source' is a path like '/data/batch_1/xxxxx.csv'. We take the
'batch_x' component (its parent directory) — NOT the CSV filename, which can
collide across batches — and look up the source directory via the batch map
(see SOURCE_TO_ADQ for the defaults; add/override batches on the command line
with --batch "<batch token>=<batch dir>", repeatable).

Destination layout merges every source batch at the subject level:
    <dst>/<Paciente>/<Estudio>/<Serie>/*
Existing files at the destination are never overwritten — a series that is
already (partly) present is topped up with only its missing files, so re-runs
and merges across batches are safe.

Usage:
    # default two batches, into the seg_model mount root
    python filter_and_copy.py [--dry-run]

    # add more batches (3rd, 4th, ...) without editing the script
    python filter_and_copy.py \
        --batch "batch_3=ADQUISICIONES-BATCH3" \
        --batch "batch_4=ADQUISICIONES-BATCH4"
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

CSV_PATH = Path("/output/CLF_performance/batches_1_5/combined_reviewed.csv")
SRC_ROOT = Path("/src/nifti")
DST_ROOT = Path("/dst/tmp_sorted_data")

EXCLUDED_W = {"Other", "DW", "Localizer", "Zip/JPG", "PD"}

# Maps each batch token (the 'batch_x' component of a row's '__source' path)
# to its batch directory under SRC_ROOT. Extend at runtime with
# --batch "<batch token>=<batch dir>" (repeatable) instead of editing this dict.
SOURCE_TO_ADQ = {
    "batch_1": "ADQUISICIONES",
    "batch_2": "ADQUISICIONES_02_03_2026",
    "batch_3": "2026-04-24",
    "batch_4": "2026-05-13",
    "batch_5": "2026-07-23",
}


def select_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    kept = [r for r in rows if (r.get("Clase W Final") or "").strip() not in EXCLUDED_W]
    return rows, kept


def extract_batch(source: str) -> str | None:
    """
    Pull the batch token from a '__source' path such as '/data/batch_1/xxx.csv'.
    Prefers a path component beginning with 'batch'; falls back to the CSV's
    immediate parent directory name. Returns None when no batch can be found.
    """
    source = (source or "").strip()
    if not source:
        return None
    parts = Path(source).parts
    for part in parts:
        if part.lower().startswith("batch"):
            return part
    # Fallback: the directory the CSV sits in (parent of the filename).
    parent = Path(source).parent.name
    return parent or None


def resolve_series_dir(row: dict, batch_map: dict[str, str],
                       src_root: Path) -> Path | None:
    """Return the source Serie directory for a row, or None if the batch is unknown."""
    batch = extract_batch(row.get("__source") or "")
    adq = batch_map.get(batch) if batch is not None else None
    if adq is None:
        return None
    return src_root / adq / row["Paciente"] / row["Estudio"] / row["Serie"]


def dest_series_dir(row: dict, dst_root: Path) -> Path:
    return dst_root / row["Paciente"] / row["Estudio"] / row["Serie"]


def copy_series(src: Path, dst: Path, dry_run: bool) -> tuple[int, int, int]:
    """
    Copy every file under src into dst (recursive), preserving directory layout.
    Returns (files_copied, bytes_copied, files_skipped). Never overwrites: any
    file that already exists at the destination is left untouched and counted
    as skipped.
    """
    if not src.exists():
        return (0, 0, 0)

    files_copied = 0
    bytes_copied = 0
    files_skipped = 0
    for item in src.rglob("*"):
        if not item.is_file():
            continue
        rel = item.relative_to(src)
        target = dst / rel
        if target.exists():
            files_skipped += 1
            continue
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
        files_copied += 1
        bytes_copied += item.stat().st_size
    return (files_copied, bytes_copied, files_skipped)


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dst", type=Path, default=DST_ROOT,
                   help=f"Destination root directory, created if missing "
                        f"(default: {DST_ROOT}).")
    p.add_argument("--src-root", type=Path, default=SRC_ROOT,
                   help=f"Root holding the batch directories (default: {SRC_ROOT}).")
    p.add_argument("--csv", type=Path, default=CSV_PATH,
                   help=f"Path to combined_reviewed.csv (default: {CSV_PATH}).")
    p.add_argument("--batch", action="append", default=[], metavar="TOKEN=DIR",
                   help="Add/override a batch mapping '<batch token>=<batch dir>' "
                        "(e.g. 'batch_3=ADQUISICIONES-BATCH3'). Repeatable — use "
                        "once per extra batch (3rd, 4th, ...).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be copied without touching the filesystem.")
    args = p.parse_args()

    if not args.csv.exists():
        print(f"ERROR: CSV not found at {args.csv}", file=sys.stderr)
        return 1

    # Build the effective batch map: defaults + any --batch overrides.
    batch_map = dict(SOURCE_TO_ADQ)
    for spec in args.batch:
        if "=" not in spec:
            print(f"ERROR: --batch must be 'TOKEN=DIR', got {spec!r}", file=sys.stderr)
            return 1
        token, batch_dir = (part.strip() for part in spec.split("=", 1))
        batch_map[token] = batch_dir
    print(f"Batches ({len(batch_map)}):")
    for token, batch_dir in batch_map.items():
        print(f"  {token}  ->  {batch_dir}")

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
        src = resolve_series_dir(r, batch_map, args.src_root)
        if src is None:
            bad_source += 1
            continue
        dst = dest_series_dir(r, args.dst)
        series_dirs.setdefault(src, dst)

    print(f"\nUnique series dirs : {len(series_dirs):,}")
    if bad_source:
        print(f"  (skipped {bad_source} rows with unknown/unmapped batch)")

    if args.dry_run:
        print("\n--dry-run: no files will be written.")

    missing = 0
    total_files = 0
    total_bytes = 0
    total_skipped = 0
    for src, dst in series_dirs.items():
        if not src.exists():
            missing += 1
            continue
        n_files, n_bytes, n_skipped = copy_series(src, dst, args.dry_run)
        total_files += n_files
        total_bytes += n_bytes
        total_skipped += n_skipped

    print(f"\nDestination root   : {args.dst}")
    print(f"Copied series dirs : {len(series_dirs) - missing:,}")
    print(f"Missing on disk    : {missing:,}")
    print(f"Files {'would be ' if args.dry_run else ''}copied : {total_files:,}")
    print(f"Bytes {'would be ' if args.dry_run else ''}copied : {fmt_bytes(total_bytes)}")
    print(f"Files skipped (already present, not overwritten): {total_skipped:,}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
