"""
Find the finest voxel spacing across all scans in a session, then resample
the best available image (T1 > T2 > others) to that isotropic-ish target.

Why: images acquired in different orientations share in-plane resolution but
have thick slices in different axes (e.g. 1×1×5 axial vs 1×5×1 sagittal).
Taking the per-axis minimum yields the finest achievable spacing (e.g. 1×1×1).
Resampling before segmentation avoids blocky edges when transferring masks back.

Directory assumption (output of dcm2nifti.py):
    <session_dir>/
        <scan_name_dir>/
            image.nii.gz
            ...

Usage:
    python high_reso_test.py <session_dir> [output.nii.gz]
"""

import re
import sys
from pathlib import Path

import SimpleITK as sitk


# ---------------------------------------------------------------------------
# Sequence-type priority (lower = preferred)
# ---------------------------------------------------------------------------

SEQ_PRIORITY = {"T1": 0, "T2": 1, "T2*": 2, "PD": 3, "DWI": 4}


def guess_sequence_type(name: str) -> str:
    u = name.upper()
    if re.search(r"T2\*", name, re.IGNORECASE):
        return "T2*"
    if re.search(r"T1", u):
        return "T1"
    if re.search(r"T2", u):
        return "T2"
    if re.search(r"DWI|DIFF|ADC", u):
        return "DWI"
    if re.search(r"PDW?", u):
        return "PD"
    return "other"


# ---------------------------------------------------------------------------
# Spacing helpers
# ---------------------------------------------------------------------------

def get_spacing(path: Path) -> tuple[float, ...]:
    """Return voxel spacing (x, y, z) in mm for a NIfTI file."""
    img = sitk.ReadImage(str(path), sitk.sitkFloat32)
    return img.GetSpacing()          # (sx, sy, sz)


def finest_spacing(spacings: list[tuple]) -> tuple[float, ...]:
    """Per-axis minimum across all images."""
    ndim = len(spacings[0])
    return tuple(min(s[i] for s in spacings) for i in range(ndim))


# ---------------------------------------------------------------------------
# Image selection
# ---------------------------------------------------------------------------

def select_best(paths: list[Path]) -> Path:
    """
    Prefer T1, then T2, then others.
    Within the same type, prefer the image with the smallest maximum spacing
    (finest worst-case axis = most isotropic / least blocky source).
    Sequence type is guessed from the scan folder name, not the file name.
    """
    def sort_key(p: Path):
        seq = guess_sequence_type(p.parent.name)   # folder = scan name
        pri = SEQ_PRIORITY.get(seq, 99)
        sp  = get_spacing(p)
        return (pri, max(sp))                      # smallest max-spacing wins

    return sorted(paths, key=sort_key)[0]


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def resample(image: sitk.Image, new_spacing: tuple[float, ...]) -> sitk.Image:
    """Resample *image* to *new_spacing* using linear interpolation."""
    old_spacing = image.GetSpacing()
    old_size    = image.GetSize()
    new_size    = [
        int(round(old_size[i] * old_spacing[i] / new_spacing[i]))
        for i in range(image.GetDimension())
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(image)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_session(session_dir: Path, output_path: Path) -> None:
    # Structure: session_dir/<scan_name_dir>/<image>.nii.gz
    niftis = sorted(session_dir.glob("*/*.nii.gz"))
    if not niftis:
        raise FileNotFoundError(f"No .nii.gz files found under {session_dir}")

    print(f"Found {len(niftis)} NIfTI file(s):")
    spacings = []
    for p in niftis:
        sp  = get_spacing(p)
        seq = guess_sequence_type(p.parent.name)   # scan folder = sequence name
        print(f"  {p.parent.name:50s}  spacing={tuple(round(s,3) for s in sp)}  seq={seq}")
        spacings.append(sp)

    target = finest_spacing(spacings)
    print(f"\nFinest spacing (per-axis min): {tuple(round(s,3) for s in target)}")

    best = select_best(niftis)
    seq  = guess_sequence_type(best.parent.name)
    sp   = get_spacing(best)
    print(f"Source image : {best.name}  (seq={seq}, spacing={tuple(round(s,3) for s in sp)})")

    image    = sitk.ReadImage(str(best), sitk.sitkFloat32)
    resampled = resample(image, target)

    print(f"Original size : {image.GetSize()}")
    print(f"Resampled size: {resampled.GetSize()}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(resampled, str(output_path))
    print(f"\nSaved → {output_path}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        sys.exit("Usage: python high_reso_test.py <session_dir> [output.nii.gz]")

    session_dir = Path(args[0])
    output_path = (
        Path(args[1]) if len(args) > 1
        else session_dir / "finest_spacing.nii.gz"
    )
    process_session(session_dir, output_path)
