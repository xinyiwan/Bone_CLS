"""
Plane <-> axis mapping and slice selection.

Assumes arrays are in canonical RAS (see volume_io.py) so the plane->axis map
is fixed. `find_max_area_slice` and `find_top_k_area_slices` are standalone and
unit-testable on any 3D mask.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

# Axis along which slices of a given plane are STACKED (RAS canonical).
PLANE_AXES = {"axial": 2, "coronal": 1, "sagittal": 0}


def _inplane_axes(plane: str) -> Tuple[int, int]:
    """The two in-plane axes (ascending) for a plane -> (row_axis, col_axis)."""
    stack = PLANE_AXES[plane]
    return tuple(a for a in (0, 1, 2) if a != stack)  # type: ignore[return-value]


def extract_slice(volume: np.ndarray, plane: str, index: int) -> np.ndarray:
    """2D slice at `index` along the plane's stacking axis. Rows/cols follow the
    ascending in-plane axis order, so spacing lookups in cropping.py line up."""
    return np.take(volume, index, axis=PLANE_AXES[plane])


def slice_areas(mask_volume: np.ndarray, plane: str) -> np.ndarray:
    """1D array: foreground voxel count per slice along the plane axis."""
    reduce_over = _inplane_axes(plane)
    return (np.asarray(mask_volume) > 0).sum(axis=reduce_over)


def find_max_area_slice(mask_volume: np.ndarray, plane: str) -> int:
    """Index of the slice with the largest tumour cross-section in `plane`.
    Returns the first max if tied. Returns argmax (0) even if the mask is empty
    -- callers should check the area is > 0 (the pipeline does)."""
    return int(np.argmax(slice_areas(mask_volume, plane)))


def find_top_k_area_slices(mask_volume: np.ndarray, plane: str, k: int = 1) -> List[int]:
    """Up to `k` slice indices ranked by area (largest first); empty slices
    (area == 0) are dropped, so the result may be shorter than k."""
    areas = slice_areas(mask_volume, plane)
    order = np.argsort(areas)[::-1]
    return [int(i) for i in order[:k] if areas[i] > 0]


def inplane_spacing(spacing: np.ndarray, plane: str) -> Tuple[float, float]:
    """(row_spacing_mm, col_spacing_mm) for a plane -- MRI is often anisotropic,
    so the two in-plane axes can have different mm/pixel."""
    r_ax, c_ax = _inplane_axes(plane)
    return float(spacing[r_ax]), float(spacing[c_ax])
