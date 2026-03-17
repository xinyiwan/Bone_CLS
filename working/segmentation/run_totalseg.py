"""
Run TotalSegmentator (total_mr task) on MR images and extract bone segmentations.

For every qualifying scan under the given root directory, this script:
  1. Runs TotalSegmentator with the 'total_mr' task, saving per-structure masks
     into  <out_dir>/<subject>/<session>/<scan>/totalseg/
  2. Merges all bone-related masks into a single multi-label NIfTI:
     <out_dir>/<subject>/<session>/<scan>/bone_seg.nii.gz
  3. Writes a JSON label map:
     <out_dir>/<subject>/<session>/<scan>/bone_seg_labels.json  {label_int: name}

The output tree mirrors the input structure but is kept entirely separate from
the source images.

Directory structure expected (output of dcm2nifti.py):
    <root>/
        <subject>/                   e.g. BONE_AI_921
            <session>/               e.g. MR_RMMUSLOS_20150129
                <scan_name>/         e.g. 4_CORT1
                    images.nii.gz

Scans whose names match SKIP_PATTERNS are silently skipped
(localisers, calibration scouts, etc.).

Usage:
    # Process every subject; outputs go to <root>_seg/ by default
    python run_totalseg.py <root_dir>

    # Specify an explicit output directory
    python run_totalseg.py <root_dir> --out_dir <out_dir>

    # Process a single session directory
    python run_totalseg.py <root_dir> --session <session_dir> --out_dir <out_dir>

    # Use fast (lower-res) model and run on GPU
    python run_totalseg.py <root_dir> [--fast] [--device gpu]
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def chmod_r(path: Path, mode: int = 0o777) -> None:
    """Recursively set permissions on *path* (file or directory tree)."""
    path.chmod(mode)
    if path.is_dir():
        for child in path.rglob("*"):
            child.chmod(mode)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_FILENAME = "images.nii.gz"


def find_scan_input(scan_dir: Path) -> Path | None:
    """
    Return the input to pass to TotalSegmentator for this scan folder:
      - scan_dir/images.nii.gz  if it exists
      - scan_dir itself         if it contains any .dcm files
      - None                    if neither is found
    """
    nifti = scan_dir / IMAGE_FILENAME
    if nifti.exists():
        return nifti
    if any(scan_dir.glob("*.dcm")):
        return scan_dir
    return None


# Scan folder name patterns to skip (case-insensitive substring match)
SKIP_PATTERNS = re.compile(
    r"localiz|calib|scout|survey|planning|loc_|_loc$|phantom|dummy",
    re.IGNORECASE,
)

# Keywords that identify bone structures in TotalSegmentator output filenames
BONE_KEYWORDS = [
    "vertebrae",
    "rib_",
    "femur",
    "hip_",
    "ilium",
    "ischium",
    "pubis",
    "sacrum",
    "coccyx",
    "humerus",
    "scapula",
    "clavicula",
    "sternum",
    "skull",
    "mandible",
    "patella",
    "tibia",
    "fibula",
    "ulna",
    "radius",
    "calcaneus",
    "talus",
    "os_",
    "carpal",
    "metacarpal",
    "metatarsal",
    "phalanx",
]


def is_bone_structure(name: str) -> bool:
    """Return True if *name* (without extension) belongs to a bone structure."""
    n = name.lower()
    return any(kw in n for kw in BONE_KEYWORDS)


def should_skip_scan(scan_name: str) -> bool:
    return bool(SKIP_PATTERNS.search(scan_name))


# ---------------------------------------------------------------------------
# TotalSegmentator wrapper
# ---------------------------------------------------------------------------

def run_totalseg(input_path: Path, output_dir: Path, fast: bool, device: str) -> bool:
    """
    Run TotalSegmentator on *input_path* with ml=True, writing a single combined
    multi-label NIfTI to output_dir/segmentations.nii.gz.
    Returns True on success, False on failure.
    """
    try:
        from totalsegmentator.python_api import totalsegmentator
    except ImportError:
        sys.exit(
            "TotalSegmentator is not installed. "
            "Install with: pip install TotalSegmentator"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        totalsegmentator(
            input=str(input_path),
            output=str(output_dir),
            task="total_mr",
            fast=fast,
            device=device,
            ml=True,        # single combined multi-label NIfTI
            preview=True,   # saves a preview PNG alongside the segmentation
            statistics=True,  # saves statistics.json with volume/intensity per structure
            verbose=False,
        )
        return True
    except Exception as exc:
        print(f"    [ERROR] TotalSegmentator failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Bone label extraction from combined multi-label output
# ---------------------------------------------------------------------------

def extract_bone_labels(
    totalseg_dir: Path,
) -> tuple[sitk.Image | None, dict[int, str]]:
    """
    Load the combined segmentations.nii.gz produced by TotalSegmentator (ml=True),
    look up TotalSegmentator's own class map to find bone class IDs, zero out
    all non-bone voxels, and return (filtered_image, {class_id: name}).
    """
    combined = totalseg_dir / "segmentations.nii.gz"
    if not combined.exists():
        print(f"    [WARN] {combined} not found")
        return None, {}

    # Resolve bone class IDs from TotalSegmentator's built-in class map
    bone_ids: dict[int, str] = {}
    try:
        from totalsegmentator.map_to_binary import class_map
        id_to_name: dict[int, str] = class_map.get("total_mr", {})
        bone_ids = {cid: name for cid, name in id_to_name.items() if is_bone_structure(name)}
    except Exception:
        pass  # fall back to keeping all labels if class map unavailable

    img = sitk.ReadImage(str(combined))
    arr = sitk.GetArrayFromImage(img).astype(np.uint16)

    if bone_ids:
        # Zero out every voxel that is not a bone structure
        bone_mask = np.isin(arr, list(bone_ids.keys()))
        arr[~bone_mask] = 0
        label_map = {int(cid): name for cid, name in bone_ids.items() if cid in np.unique(arr)}
    else:
        # Class map unavailable — keep the full combined segmentation as-is
        label_map = {int(v): str(v) for v in np.unique(arr) if v != 0}

    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img)
    return out, label_map


# ---------------------------------------------------------------------------
# Per-scan processing
# ---------------------------------------------------------------------------

def process_scan(scan_dir: Path, out_scan_dir: Path, fast: bool, device: str) -> None:
    image_path = find_scan_input(scan_dir)
    if image_path is None:
        print(f"  [SKIP] {scan_dir.name}  (no images.nii.gz or .dcm files)")
        return

    if should_skip_scan(scan_dir.name):
        print(f"  [SKIP] {scan_dir.name}  (matches skip pattern)")
        return

    bone_out = out_scan_dir / "bone_seg.nii.gz"
    if bone_out.exists():
        print(f"  [DONE] {scan_dir.name}  (bone_seg.nii.gz already exists)")
        return

    print(f"  [RUN ] {scan_dir.name}")

    totalseg_dir = out_scan_dir / "totalseg"
    success = run_totalseg(image_path, totalseg_dir, fast=fast, device=device)
    if not success:
        return
    chmod_r(totalseg_dir)

    merged, label_map = extract_bone_labels(totalseg_dir)
    if merged is None:
        print(f"    [WARN] no bone structures found in TotalSegmentator output")
        return

    out_scan_dir.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(merged, str(bone_out))
    bone_out.chmod(0o777)

    label_json = out_scan_dir / "bone_seg_labels.json"
    with open(label_json, "w") as fh:
        json.dump(label_map, fh, indent=2)
    label_json.chmod(0o777)

    chmod_r(out_scan_dir)   # ensure all parent dirs in the output tree are accessible

    print(
        f"    saved {len(label_map)} bone labels  →  {bone_out.relative_to(out_scan_dir.parent.parent)}"
    )


# ---------------------------------------------------------------------------
# Session / root iteration
# ---------------------------------------------------------------------------

def process_session(
    session_dir: Path, out_session_dir: Path, fast: bool, device: str
) -> None:
    """Process all scans within a single session directory."""
    scan_dirs = sorted(
        p for p in session_dir.iterdir()
        if p.is_dir() and find_scan_input(p) is not None
    )
    if not scan_dirs:
        print(f"  [SKIP] session {session_dir.name}  (no scan dirs with images.nii.gz or .dcm files)")
        return

    print(f"\n{'='*60}")
    print(f"Session : {session_dir}")
    print(f"Out     : {out_session_dir}")
    print(f"{'='*60}")
    for scan_dir in scan_dirs:
        out_scan_dir = out_session_dir / scan_dir.name
        process_scan(scan_dir, out_scan_dir, fast=fast, device=device)


def process_root(root_dir: Path, out_dir: Path, fast: bool, device: str) -> None:
    """
    Walk root / subject / session / scan, mirroring the structure under out_dir.
    """
    session_dirs = sorted(
        p
        for subject in sorted(root_dir.iterdir()) if subject.is_dir()
        for p in sorted(subject.iterdir()) if p.is_dir()
    )
    if not session_dirs:
        # root is itself a session dir
        process_session(root_dir, out_dir, fast=fast, device=device)
        return

    for session_dir in session_dirs:
        rel = session_dir.relative_to(root_dir)
        process_session(session_dir, out_dir / rel, fast=fast, device=device)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root_dir", type=Path, help="Root directory containing subject folders")
    p.add_argument(
        "--out_dir", type=Path, default=None,
        help="Output root (default: <root_dir>_seg next to root_dir)",
    )
    p.add_argument(
        "--session", type=Path, default=None,
        help="Process a single session directory instead of walking the whole root",
    )
    p.add_argument(
        "--fast", action="store_true",
        help="Use TotalSegmentator fast (lower-resolution) model",
    )
    p.add_argument(
        "--device", default="cpu", choices=["cpu", "gpu", "mps"],
        help="Inference device (default: cpu)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir: Path = args.out_dir or args.root_dir.parent / (args.root_dir.name + "_seg")

    start = datetime.now()
    print(f"Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input  : {args.root_dir}")
    print(f"Output : {out_dir}")
    print(f"Task   : total_mr  |  fast={args.fast}  |  device={args.device}")

    if args.session:
        # Mirror the session's relative position under out_dir if possible
        try:
            rel = args.session.relative_to(args.root_dir)
            out_session_dir = out_dir / rel
        except ValueError:
            out_session_dir = out_dir / args.session.name
        process_session(args.session, out_session_dir, fast=args.fast, device=args.device)
    else:
        process_root(args.root_dir, out_dir, fast=args.fast, device=args.device)

    elapsed = datetime.now() - start
    print(f"\nFinished in {elapsed}")


if __name__ == "__main__":
    main()
