"""
Downsample the FINAL segmentation back to the source scan's native space,
then register it to every other scan in the session.

Steps:
  1. Find FINAL_<scan_name>_<timestamp>.nii.gz  (made on one upsampled scan)
  2. Downsample it to the source scan's original spacing (nearest-neighbour)
  3. For every other scan in the session:
       a. Register source image → target image (rigid, mutual information)
       b. Apply the transform to the downsampled segmentation (nearest-neighbour)
       c. Save alongside the target scan

Expected structure:
    <session_dir>/
        <scan_name>/
            images.nii.gz
        resampled/
            segmentation_history/
                FINAL_<scan_name>_<YYYYMMDD>_<HHMMSS>.nii.gz

Output:
    <session_dir>/
        <scan_name>/                    ← source scan
            FINAL_<scan_name>_<ts>.nii.gz
        <other_scan>/                   ← every other scan
            FINAL_<scan_name>_<ts>.nii.gz

Usage:
    python downsample_seg.py <session_dir>
"""

import os
import re
import sys
from pathlib import Path

import SimpleITK as sitk


FINAL_PATTERN = re.compile(r"^FINAL_(.+)_(\d{8}_\d{6})\.nii\.gz$")
ORIGINAL_IMAGE = "images.nii.gz"


# ---------------------------------------------------------------------------
# Resampling helpers
# ---------------------------------------------------------------------------

def resample_to_reference(
    moving: sitk.Image,
    ref: sitk.Image,
    interpolator=sitk.sitkNearestNeighbor,
    transform: sitk.Transform | None = None,
) -> sitk.Image:
    """Resample *moving* onto the grid of *ref*, optionally applying *transform*."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(ref.GetSpacing())
    resampler.SetSize(ref.GetSize())
    resampler.SetOutputDirection(ref.GetDirection())
    resampler.SetOutputOrigin(ref.GetOrigin())
    resampler.SetTransform(transform if transform is not None else sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(interpolator)
    return resampler.Execute(moving)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_rigid(fixed: sitk.Image, moving: sitk.Image) -> sitk.Transform:
    """
    Rigid registration of *moving* onto *fixed*.
    Returns the transform that maps fixed → moving (for resampling moving into
    fixed space via ResampleImageFilter, which uses the inverse internally).
    """
    fixed_f  = sitk.Cast(sitk.RescaleIntensity(fixed),  sitk.sitkFloat32)
    moving_f = sitk.Cast(sitk.RescaleIntensity(moving), sitk.sitkFloat32)

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.1)
    reg.SetInterpolator(sitk.sitkLinear)

    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=200,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    reg.SetOptimizerScalesFromPhysicalShift()

    tx = sitk.CenteredTransformInitializer(
        fixed_f, moving_f,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    reg.SetInitialTransform(tx, inPlace=False)

    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    return reg.Execute(fixed_f, moving_f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_session(session_dir: Path) -> None:
    seg_dir = session_dir / "resampled" / "segmentation_history"
    if not seg_dir.exists():
        raise FileNotFoundError(f"Segmentation history dir not found: {seg_dir}")

    finals = sorted(seg_dir.glob("FINAL_*.nii.gz"))
    if not finals:
        raise FileNotFoundError(f"No FINAL_*.nii.gz found in {seg_dir}")

    if len(finals) > 1:
        print(f"Warning: {len(finals)} FINAL files found; using the first one.")

    seg_path = finals[0]
    m = FINAL_PATTERN.match(seg_path.name)
    if not m:
        sys.exit(f"Cannot parse scan name from: {seg_path.name}")

    source_scan = m.group(1)
    source_ref  = session_dir / source_scan / ORIGINAL_IMAGE
    if not source_ref.exists():
        sys.exit(f"Source reference image not found: {source_ref}")

    print(f"FINAL seg : {seg_path.name}")
    print(f"Source scan: {source_scan}")

    # --- Step 1: downsample seg to source scan's native space ----------------
    seg     = sitk.ReadImage(str(seg_path))
    src_img = sitk.ReadImage(str(source_ref))
    seg_native = resample_to_reference(seg, src_img)

    out_source = session_dir / source_scan / seg_path.name
    sitk.WriteImage(seg_native, str(out_source))
    os.chmod(out_source, 0o666)
    print(f"  [{source_scan}] downsampled  {seg.GetSize()} → {seg_native.GetSize()}  saved → {out_source}")

    # --- Step 2: register seg to every other scan ----------------------------
    other_scans = [
        p.parent for p in sorted(session_dir.glob(f"*/{ORIGINAL_IMAGE}"))
        if p.parent.name != source_scan
        and p.parent.name != "resampled"
    ]

    print(f"\nRegistering to {len(other_scans)} other scan(s):")
    for scan_dir in other_scans:
        tgt_path = scan_dir / ORIGINAL_IMAGE
        tgt_img  = sitk.ReadImage(str(tgt_path))

        print(f"  [{scan_dir.name}] registering ...", end=" ", flush=True)
        transform = register_rigid(fixed=tgt_img, moving=src_img)

        seg_registered = resample_to_reference(
            seg_native, tgt_img,
            interpolator=sitk.sitkNearestNeighbor,
            transform=transform,
        )

        out_path = scan_dir / seg_path.name
        sitk.WriteImage(seg_registered, str(out_path))
        os.chmod(out_path, 0o666)
        print(f"done  {seg_registered.GetSize()}  saved → {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python downsample_seg.py <session_dir>")
    process_session(Path(sys.argv[1]))
