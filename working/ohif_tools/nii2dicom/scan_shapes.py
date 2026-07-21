"""
Scan a NIfTI dataset and report the array shape / dimensionality of every image,
so you know up front which series are non-3-D (4-D multi-echo/Dixon, or odd cases)
before running convert.py on a large dataset.

Only the NIfTI *header* is read (not the pixel data), so it's fast even for
thousands of files.

USAGE
-----
    python scan_shapes.py --input /path/to/filtered_nifti
    python scan_shapes.py --input /path/to/filtered_nifti --pattern '*.nii.gz'
    python scan_shapes.py --input /path/to/filtered_nifti --report shapes_report.csv

By default it scans files named 'images.nii.gz'. Use --pattern '*.nii.gz' to scan
every NIfTI (e.g. to include segmentations too).

Requires: nibabel  (pip install nibabel)
"""
import os
import csv
import argparse
import fnmatch
from collections import Counter

import nibabel as nib


def find_files(root, pattern):
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fnmatch.fnmatch(fn, pattern):
                yield os.path.join(dirpath, fn)


def main():
    ap = argparse.ArgumentParser(description="Report NIfTI shapes/dimensionality across a dataset.")
    ap.add_argument("--input", required=True, help="dataset root to scan")
    ap.add_argument("--pattern", default="images.nii.gz",
                    help="filename glob to match (default: images.nii.gz; use '*.nii.gz' for all)")
    ap.add_argument("--report", default="shapes_report.csv",
                    help="CSV output path (default: shapes_report.csv)")
    args = ap.parse_args()

    root = os.path.abspath(args.input)
    files = sorted(find_files(root, args.pattern))
    print(f"Scanning {len(files)} file(s) matching '{args.pattern}' under {root}\n")

    rows = []
    ndim_counts = Counter()
    vol_counts = Counter()   # for 4-D: how many volumes (T)
    non3d = []
    errors = []

    for i, path in enumerate(files):
        rel = os.path.relpath(path, root)
        try:
            shape = tuple(int(x) for x in nib.load(path).header.get_data_shape())
            ndim = len(shape)
            ndim_counts[ndim] += 1
            note = "ok" if ndim == 3 else ("4D-multivolume" if ndim == 4 else "UNEXPECTED-NDIM")
            if ndim == 4:
                vol_counts[shape[3]] += 1
            if ndim != 3:
                non3d.append((rel, shape))
            rows.append([rel, ndim, "x".join(map(str, shape)), note])
        except Exception as exc:
            errors.append((rel, str(exc)))
            rows.append([rel, "", "", f"ERROR: {exc}"])
        if (i + 1) % 500 == 0:
            print(f"  ...{i+1}/{len(files)}")

    with open(args.report, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "ndim", "shape", "note"])
        w.writerows(rows)

    # ---- summary ----
    print("\n===== SUMMARY =====")
    print(f"total files scanned : {len(files)}")
    for nd in sorted(ndim_counts):
        label = {3: "3-D (normal)"}.get(nd, f"{nd}-D")
        print(f"  {label:16s}: {ndim_counts[nd]}")
    if vol_counts:
        print("  4-D volume counts (T):", dict(sorted(vol_counts.items())))
    if errors:
        print(f"  unreadable files   : {len(errors)}")

    if non3d:
        print(f"\n----- {len(non3d)} non-3-D file(s) -----")
        for rel, shape in non3d[:50]:
            print(f"  {'x'.join(map(str, shape)):>20s}  {rel}")
        if len(non3d) > 50:
            print(f"  ... and {len(non3d) - 50} more (see {args.report})")
    else:
        print("\nAll matched files are 3-D. convert.py will handle them directly.")

    if errors:
        print(f"\n----- {len(errors)} unreadable file(s) -----")
        for rel, msg in errors[:20]:
            print(f"  {rel}: {msg}")

    print(f"\nFull report written to: {os.path.abspath(args.report)}")


if __name__ == "__main__":
    main()
