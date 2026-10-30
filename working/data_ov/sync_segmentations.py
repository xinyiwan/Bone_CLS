"""Sync segmentation_history/ folders made under a subset copy back into the
main sorted-data tree, one session at a time (rsync -- additive, never
deletes anything on the destination).

Run locally.
"""

import argparse
import subprocess
from pathlib import Path

DEFAULT_DST_ROOT = Path("/Volumes/SanDisk/BONE-AI/tmp_sorted_data")
SEG_HISTORY_DIRNAME = "segmentation_history"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("subset_dir", type=Path,
                    help="subset copy to sync from, e.g. output/subset_data/20pct")
    ap.add_argument("--dst-root", type=Path, default=DEFAULT_DST_ROOT)
    ap.add_argument("--dry-run", action="store_true", help="show what would sync, change nothing")
    ap.add_argument("--checksum", action="store_true",
                    help="skip unchanged files by content hash instead of size+mtime")
    args = ap.parse_args()

    seg_dirs = sorted(args.subset_dir.glob(f"*/*/{SEG_HISTORY_DIRNAME}"))
    if not seg_dirs:
        raise SystemExit(f"No {SEG_HISTORY_DIRNAME}/ folders found under {args.subset_dir}")

    synced = 0
    for src in seg_dirs:
        session_rel = src.relative_to(args.subset_dir).parent  # <subject>/<session>
        dst = args.dst_root / session_rel / SEG_HISTORY_DIRNAME
        dst.mkdir(parents=True, exist_ok=True)
        cmd = (["rsync", "-av"]
               + (["--dry-run"] if args.dry_run else [])
               + (["--checksum"] if args.checksum else [])
               + [f"{src}/", f"{dst}/"])
        print(f"-- {session_rel} --")
        subprocess.run(cmd, check=True)
        synced += 1

    print(f"\nSynced {synced} session(s) -> {args.dst_root}")


if __name__ == "__main__":
    main()
