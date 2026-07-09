"""
NIfTI I/O with EXPLICIT orientation handling.

We use nibabel (dominant in this repo) and reorient every volume + mask to
canonical RAS with `nib.as_closest_canonical`. This means downstream code can
rely on a FIXED axis order regardless of how each scanner stored the data:

    axis 0 -> R (left..right)   -> sagittal slices stack along here
    axis 1 -> A (post..ant)     -> coronal  slices stack along here
    axis 2 -> S (inf..sup)      -> axial    slices stack along here

IMPORTANT ASSUMPTION: the mask is assumed to live on the SAME voxel grid /
affine as the volume it segments (already co-registered/resampled). We do NOT
resample here. We *do* check shape and affine and warn loudly on mismatch --
if you start seeing affine warnings, add a resampling step (e.g.
nibabel.processing.resample_from_to) before trusting the crops.

CAVEAT to double-check on YOUR data: `as_closest_canonical` snaps to the
closest axis-aligned orientation. For oblique acquisitions the plane you get is
only *approximately* axial/coronal/sagittal. Before trusting a whole dataset,
eyeball a few overlays (see qc_contact_sheet.py / overlay mode) across scanners.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import nibabel as nib


def voxel_spacing(affine: np.ndarray) -> np.ndarray:
    """Per-axis voxel size in mm, derived from the affine (order matches axes)."""
    m = np.asarray(affine)[:3, :3]
    return np.sqrt((m ** 2).sum(axis=0))


def load_canonical(path: str | Path) -> nib.Nifti1Image:
    return nib.as_closest_canonical(nib.load(str(path)))


def load_volume_and_mask(
    vol_path: str | Path, mask_path: str | Path
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (volume float32, mask int, spacing[3] mm, affine 4x4), both arrays
    reoriented to canonical RAS. Raises FileNotFoundError if either is missing.
    """
    vol_path, mask_path = Path(vol_path), Path(mask_path)
    if not vol_path.exists():
        raise FileNotFoundError(vol_path)
    if not mask_path.exists():
        raise FileNotFoundError(mask_path)

    vimg = load_canonical(vol_path)
    mimg = load_canonical(mask_path)

    vol = np.asarray(vimg.get_fdata(dtype=np.float32))
    mask = np.asanyarray(mimg.dataobj)  # keep integer labels, no float rounding

    if mask.shape[:3] != vol.shape[:3]:
        raise ValueError(
            f"Shape mismatch after reorientation: vol {vol.shape} vs mask "
            f"{mask.shape} ({vol_path.name}). Grids differ -- add resampling."
        )
    if not np.allclose(vimg.affine, mimg.affine, atol=1e-3):
        # Same shape, different affine = the silently-misaligned case flagged in
        # qc_overlay.py. We keep going but warn; add resampling if you see this.
        warnings.warn(
            f"Affine mismatch (same shape) for {vol_path.name}: mask may be "
            f"voxel-misaligned. Verify before trusting crops.",
            stacklevel=2,
        )

    return vol, mask.astype(np.int16), voxel_spacing(vimg.affine), vimg.affine
