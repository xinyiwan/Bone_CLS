"""
Find the finest voxel spacing across all scans in a session, then resample
every image to that isotropic-ish target and save them to an output directory.

Why: images acquired in different orientations share in-plane resolution but
have thick slices in different axes (e.g. 1×1×5 axial vs 1×5×1 sagittal).
Taking the per-axis minimum yields the finest achievable spacing (e.g. 1×1×1).
Resampling before segmentation avoids blocky edges when transferring masks back.

Directory assumption (output of dcm2nifti.py):
    <session_dir>/
        <scan_name_dir>/
            image.nii.gz
            ...

Output mirrors the input structure under <output_dir>:
    <output_dir>/
        <scan_name_dir>/
            image.nii.gz
            ...

Usage:
    python high_reso_test.py <session_dir> [output_dir]
"""

import re
import sys
from pathlib import Path

import SimpleITK as sitk


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

def process_session(session_dir: Path, output_dir: Path) -> None:
    # Structure: session_dir/<scan_name_dir>/<image>.nii.gz
    niftis = sorted(session_dir.glob("*/*.nii.gz"))
    if not niftis:
        raise FileNotFoundError(f"No .nii.gz files found under {session_dir}")

    print(f"Found {len(niftis)} NIfTI file(s):")
    spacings = []
    for p in niftis:
        sp  = get_spacing(p)
        seq = guess_sequence_type(p.parent.name)
        print(f"  {p.parent.name:50s}  spacing={tuple(round(s,3) for s in sp)}  seq={seq}")
        spacings.append(sp)

    target = finest_spacing(spacings)
    print(f"\nFinest spacing (per-axis min): {tuple(round(s,3) for s in target)}")

    for p in niftis:
        image     = sitk.ReadImage(str(p), sitk.sitkFloat32)
        resampled = resample(image, target)

        # Mirror input structure: output_dir/<scan_name_dir>/<filename>
        rel       = p.relative_to(session_dir)
        out_path  = output_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(resampled, str(out_path))
        print(f"  {p.parent.name}/{p.name}  {image.GetSize()} → {resampled.GetSize()}  saved → {out_path}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        sys.exit("Usage: python high_reso_test.py <session_dir> [output_dir]")

    session_dir = Path(args[0])
    output_dir  = Path(args[1]) if len(args) > 1 else session_dir / "resampled"
    process_session(session_dir, output_dir)
