"""
Per-volume intensity normalization.

MRI intensities are arbitrary (no Hounsfield-like scale), so we normalize each
volume before cropping. Kept as a single swappable function.

Steps:
  1. Compute low/high percentiles (default 1st/99th) -- ideally on FOREGROUND
     voxels only, so a large air/background FOV doesn't drag the percentiles.
     `foreground_only=True` uses non-zero voxels as a cheap foreground proxy.
  2. Clip to [low, high].
  3. Rescale: 'minmax' -> out_range, or 'zscore' -> mean/std then squashed into
     out_range via a +/- z clip (so PNG export has a fixed range).

Returns float32 in `out_range` (default [0, 1]); the save step scales to 8-bit.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def normalize_intensity(
    volume: np.ndarray,
    method: str = "minmax",
    p_low: float = 1.0,
    p_high: float = 99.0,
    out_range: Tuple[float, float] = (0.0, 1.0),
    foreground_only: bool = True,
    z_clip: float = 3.0,
) -> np.ndarray:
    vol = np.asarray(volume, dtype=np.float32)

    sample = vol[vol != 0] if foreground_only else vol.ravel()
    if sample.size == 0:  # empty/all-zero volume: nothing to scale
        sample = vol.ravel()

    lo, hi = np.percentile(sample, [p_low, p_high])
    if hi <= lo:  # flat volume; avoid divide-by-zero
        hi = lo + 1.0

    clipped = np.clip(vol, lo, hi)
    out_lo, out_hi = out_range

    if method == "minmax":
        norm = (clipped - lo) / (hi - lo)
    elif method == "zscore":
        fg = clipped[vol != 0] if foreground_only else clipped
        mean, std = float(fg.mean()), float(fg.std())
        std = std if std > 0 else 1.0
        z = (clipped - mean) / std
        norm = (np.clip(z, -z_clip, z_clip) + z_clip) / (2 * z_clip)
    else:
        raise ValueError(f"Unknown normalization method: {method!r}")

    return (norm * (out_hi - out_lo) + out_lo).astype(np.float32)
