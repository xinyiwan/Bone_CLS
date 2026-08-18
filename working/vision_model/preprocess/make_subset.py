"""
Build a small, server-shippable copy of the dataset containing only the scans
that actually have a segmentation (history or reviewed).

Discovery is the exact same segmentation-driven walk run.py uses
(seg_model/pairs.find_pairs), so whatever this copies is precisely what run.py
will later find under the new root. The on-disk layout is preserved:

    <out-root>/<subject>/<session>/<scan>/images.nii.gz
    <out-root>/<subject>/<session>/segmentation_history/segs/<scan>_seg.nii.gz
    <out-root>/<subject>/<session>/review/<xxx>/segs/<scan>_seg.nii.gz

Only the *resolved* mask for each (session, scan) is copied -- the one
resolve_seg would pick (reviewed wins over history) -- at its original relative
path, so resolution on the server yields the same pair and the same `source`.

Usage:
  # see what would be copied + the total size, without touching anything
  python make_subset.py --data-root /Volumes/SanDisk/BONE-AI/tmp_sorted_data \\
      --out-root ./subset --dry-run

  # copy it
  python make_subset.py --data-root /Volumes/SanDisk/BONE-AI/tmp_sorted_data \\
      --out-root ./subset

  # only radiologist-reviewed masks / only some subjects
  python make_subset.py ... --reviewed-only --subjects SUBJ001 SUBJ002

Then rsync ./subset to the server and point run.py at it:
  rsync -avP ./subset/ user@server:/data/bone_subset/
  python run.py --data-root /data/bone_subset --out-root ./out ...
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

# Reuse the segmentation project's discovery, same as run.py does.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "working" / "seg_model"))


def human(n: int) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024 or unit == "TB":
            return f"{x:.1f}{unit}"
        x /= 1024
    return f"{x:.1f}TB"


def copy_one(src: Path, dst: Path, mode: str, dry_run: bool) -> int:
    """Copy/link src -> dst, creating parents. Returns bytes accounted for."""
    size = src.stat().st_size
    if dry_run:
        return size
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return size
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "hardlink":
        try:
            dst.hardlink_to(src)
        except OSError:  # cross-device -> fall back to a real copy
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)
    return size


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-root", type=Path, required=True, help="full dataset root")
    ap.add_argument("--out-root", type=Path, required=True, help="where to build the subset")
    ap.add_argument("--subjects", nargs="*", help="restrict to these subject ids")
    ap.add_argument("--reviewed-only", action="store_true",
                    help="keep only radiologist-reviewed masks (matches run.py --reviewed-only)")
    ap.add_argument("--mode", choices=["copy", "symlink", "hardlink"], default="copy",
                    help="copy (default; what you want before rsync), or link for a local scratch subset")
    ap.add_argument("--scan-extra", nargs="*", default=[],
                    help="extra filename globs to take from each kept scan folder, e.g. '*.json'")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    from pairs import find_pairs  # lazy: needs nibabel

    keep_subjects = set(args.subjects) if args.subjects else None
    rows, total = [], 0
    n_no_image = n_skipped = 0

    for subject, session, scan, image_path, seg_path, source in find_pairs(args.data_root):
        if keep_subjects is not None and subject not in keep_subjects:
            n_skipped += 1
            continue
        if args.reviewed_only and source != "reviewed":
            n_skipped += 1
            continue
        if image_path is None:  # mask whose image was excluded/removed
            n_no_image += 1
            continue

        # Preserve each file's path relative to the original root.
        pairs_to_copy = [image_path, seg_path]
        for name in args.scan_extra:
            pairs_to_copy.extend(sorted(image_path.parent.glob(name)))

        n_bytes = 0
        for src in pairs_to_copy:
            dst = args.out_root / src.relative_to(args.data_root)
            n_bytes += copy_one(src, dst, args.mode, args.dry_run)
        total += n_bytes

        rows.append({
            "subject": subject, "session": session, "scan": scan, "source": source,
            "image_rel": str(image_path.relative_to(args.data_root)),
            "seg_rel": str(seg_path.relative_to(args.data_root)),
            "bytes": n_bytes,
        })
        print(f"[{len(rows):5d}] {subject}/{session}/{scan}  ({source}, {human(n_bytes)})")

    if not rows:
        raise SystemExit("nothing matched -- check --data-root / --subjects / --reviewed-only")

    df = pd.DataFrame(rows)
    n_subj = df["subject"].nunique()
    n_sess = df[["subject", "session"]].drop_duplicates().shape[0]
    print(f"\n{len(df)} scan(s), {n_sess} session(s), {n_subj} subject(s), {human(total)} total"
          f"  [{n_no_image} mask(s) had no image, {n_skipped} filtered out]")
    print(df["source"].value_counts().to_string())

    if args.dry_run:
        print("\n(dry run -- nothing written)")
        return

    manifest = args.out_root / "subset_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest, index=False)
    print(f"\nwrote {manifest}")
    print(f"subset root: {args.out_root.resolve()}")


if __name__ == "__main__":
    main()
