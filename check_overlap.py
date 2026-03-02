"""
Check overlapping subjects between two DICOM data directories.

Usage:
    python check_overlap.py /path/to/DATADIR1 /path/to/DATADIR2
"""

import sys
from pathlib import Path


def get_subjects(datadir: Path) -> set[str]:
    return {p.name for p in datadir.iterdir() if p.is_dir()}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python check_overlap.py DATADIR1 DATADIR2")

    dir1, dir2 = Path(sys.argv[1]), Path(sys.argv[2])

    subjects1 = get_subjects(dir1)
    subjects2 = get_subjects(dir2)
    overlap    = subjects1 & subjects2

    print(f"Subjects in {dir1.name}: {len(subjects1)}")
    print(f"Subjects in {dir2.name}: {len(subjects2)}")
    print(f"Overlapping: {len(overlap)}")
    if overlap:
        for s in sorted(overlap):
            print(f"  {s}")
