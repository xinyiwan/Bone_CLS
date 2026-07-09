"""
2D bounding-box cropping around a mask, with a configurable margin.

Margin can be a fixed pixel count OR a real-world distance in mm (converted with
the two in-plane voxel spacings -- which can differ, since MRI is anisotropic).

Boundary handling is configurable:
  - 'clip' (default): shrink the box to the image edge; output smaller than the
     requested extent near borders.
  - 'pad':  keep the requested extent and zero-pad outside the image.

All bboxes are (row_min, row_max, col_min, col_max), INCLUSIVE.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

BBox = Tuple[int, int, int, int]


def mask_bbox_2d(mask2d: np.ndarray) -> Optional[BBox]:
    """Tight inclusive bbox of foreground, or None if the slice mask is empty."""
    ys, xs = np.where(np.asarray(mask2d) > 0)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def margin_px_from_mm(margin_mm: float, row_spacing: float, col_spacing: float) -> Tuple[int, int]:
    """Convert a mm margin to per-axis pixel margins (rounded)."""
    return int(round(margin_mm / row_spacing)), int(round(margin_mm / col_spacing))


def expand_bbox(bbox: BBox, margin_rows: int, margin_cols: int) -> BBox:
    """Grow the box by the given per-axis pixel margins. May go out of bounds;
    crop_with_bbox resolves that per the chosen mode (so the *requested* extent
    is preserved in metadata)."""
    r0, r1, c0, c1 = bbox
    return r0 - margin_rows, r1 + margin_rows, c0 - margin_cols, c1 + margin_cols


def crop_with_bbox(arr: np.ndarray, bbox: BBox, mode: str = "clip", pad_value: float = 0.0) -> np.ndarray:
    """Crop `arr` (2D) to `bbox`. 'clip' shrinks to image bounds; 'pad' keeps the
    full requested size and fills outside with `pad_value`."""
    arr = np.asarray(arr)
    h, w = arr.shape
    r0, r1, c0, c1 = bbox

    if mode == "clip":
        r0, r1 = max(0, r0), min(h - 1, r1)
        c0, c1 = max(0, c0), min(w - 1, c1)
        return arr[r0 : r1 + 1, c0 : c1 + 1]

    if mode == "pad":
        out = np.full((r1 - r0 + 1, c1 - c0 + 1), pad_value, dtype=arr.dtype)
        sr0, sr1 = max(0, r0), min(h - 1, r1)
        sc0, sc1 = max(0, c0), min(w - 1, c1)
        if sr1 >= sr0 and sc1 >= sc0:  # some overlap with the image
            out[sr0 - r0 : sr1 - r0 + 1, sc0 - c0 : sc1 - c0 + 1] = arr[sr0 : sr1 + 1, sc0 : sc1 + 1]
        return out

    raise ValueError(f"Unknown crop mode: {mode!r} (use 'clip' or 'pad')")


def apply_mask(
    img2d: np.ndarray, mask2d: np.ndarray, background: float = 0.0, dilate_px: int = 0
) -> np.ndarray:
    """Zero out (set to `background`) everything OUTSIDE the segmentation, keeping
    only the lesion shape. `dilate_px` grows the kept region by a few pixels so
    the informative peritumoral rim (e.g. enhancement) isn't clipped.

    Note: this is applied at crop resolution BEFORE resizing, so the subsequent
    bilinear resize will feather the lesion edge into the background -- expected.
    """
    m = (np.asarray(mask2d) > 0).astype(np.uint8)
    if dilate_px > 0:
        import cv2  # local import: only masked mode needs it

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        m = cv2.dilate(m, k)
    out = np.asarray(img2d, dtype=np.float32).copy()
    out[m == 0] = background
    return out
