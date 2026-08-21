"""
Box jitter, overlap metrics, and contour-roughness descriptors.

WHY ROUGHNESS AND NOT JUST DICE
    Dice asks "does the predicted region cover the same pixels". It is almost
    blind to the property this project actually extracts: the CHARACTER of the
    margin. A segmenter with a strong smoothness prior (every SAM descendant has
    one) can score Dice 0.85 while replacing a lobulated, spiculated boundary
    with a smooth blob -- and the `shape` feature in medgemma_pilot is then being
    read off a curve the segmenter invented. Roughness is the metric that catches
    that, so every comparison here reports the two side by side.

    The radial-harmonic descriptors are deliberately the SAME parameterisation as
    the synthetic families in shape_probe/shapes.py:

        r(theta) / mean(r) = 1 + sum_k amp_k * sin(k*theta + phase_k)

    so `lobe_amp` (k=3-7) and `jag_amp` (k=8-22) measured on a real lesion are
    directly comparable to the `--difficulty` axis the shape probe sweeps. That
    is what turns "the model got 60% on shape" into "real lesions sit at
    deformation amplitude ~0.12 and the model's threshold is ~0.15".

CAVEAT on the radial parameterisation
    r(theta) assumes the contour is star-shaped about its centroid. Real lesions
    mostly are; strongly exophytic or crescent ones are not, and for those we
    take the OUTERMOST crossing per angle bin. That is a documented
    approximation, not an exact shape descriptor -- it under-reports concavity.
    `solidity` does not make the assumption, which is why it is reported too.
"""

from __future__ import annotations

import math
import random
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

N_ANGLES = 512          # radial resampling resolution
LOBE_K = (3, 7)         # "several rounded lobes"   -> lobulated
JAG_K = (8, 22)         # "many small projections"  -> irregular


# --------------------------------------------------------------------------
# boxes
# --------------------------------------------------------------------------

def tight_box(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """(x0, y0, x1, y1) inclusive, or None for an empty mask."""
    ys, xs = np.nonzero(np.asarray(mask) > 0)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def jitter_box(
    box: Tuple[int, int, int, int],
    shape: Tuple[int, int],
    rng: random.Random,
    shift_frac: float = 0.10,
    scale_range: Tuple[float, float] = (0.90, 1.20),
) -> Tuple[int, int, int, int]:
    """Randomly translate and rescale a box, clipped to the image.

    The tight box of the ground-truth mask encodes the lesion's extent to the
    pixel. Prompt SAM with it and SAM snaps to it, so you measure "can it fill in
    a box derived from the answer" -- which flatters it badly and tells you
    nothing about a deployable pipeline, where the box comes from a detector or a
    rough click. Jitter is what makes the number mean something: if Dice holds up
    under it the segmenter is finding the lesion, and sweeping `shift_frac` tells
    you how accurate an upstream detector would have to be.

    Defaults (+-10% of box size, 0.9-1.2x) are deliberately mild -- roughly a
    good detector. Sweep upward to find the breaking point."""
    h, w = shape
    x0, y0, x1, y1 = box
    bw, bh = (x1 - x0 + 1), (y1 - y0 + 1)
    cx, cy = x0 + bw / 2.0, y0 + bh / 2.0

    cx += rng.uniform(-shift_frac, shift_frac) * bw
    cy += rng.uniform(-shift_frac, shift_frac) * bh
    bw *= rng.uniform(*scale_range)
    bh *= rng.uniform(*scale_range)

    nx0 = int(round(max(0, cx - bw / 2.0)))
    ny0 = int(round(max(0, cy - bh / 2.0)))
    nx1 = int(round(min(w - 1, cx + bw / 2.0)))
    ny1 = int(round(min(h - 1, cy + bh / 2.0)))
    # A degenerate box after clipping would make the prompt meaningless.
    if nx1 <= nx0:
        nx0, nx1 = max(0, min(nx0, w - 2)), min(w - 1, max(nx0 + 1, nx1))
    if ny1 <= ny0:
        ny0, ny1 = max(0, min(ny0, h - 2)), min(h - 1, max(ny0 + 1, ny1))
    return nx0, ny0, nx1, ny1


def box_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Recorded per row so a bad Dice can be traced to a bad prompt."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = max(0, min(ax1, bx1) - max(ax0, bx0) + 1)
    iy = max(0, min(ay1, by1) - max(ay0, by0) + 1)
    inter = ix * iy
    area_a = (ax1 - ax0 + 1) * (ay1 - ay0 + 1)
    area_b = (bx1 - bx0 + 1) * (by1 - by0 + 1)
    union = area_a + area_b - inter
    return float(inter) / union if union > 0 else 0.0


# --------------------------------------------------------------------------
# overlap
# --------------------------------------------------------------------------

def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a) > 0
    b = np.asarray(b) > 0
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2.0 * (a & b).sum()) / s


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a) > 0
    b = np.asarray(b) > 0
    u = (a | b).sum()
    return 1.0 if u == 0 else float((a & b).sum()) / u


def _boundary_points(mask: np.ndarray) -> np.ndarray:
    cnts, _ = cv2.findContours((np.asarray(mask) > 0).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return np.empty((0, 2), dtype=np.float64)
    return np.vstack([c.reshape(-1, 2) for c in cnts]).astype(np.float64)


def hd95(a: np.ndarray, b: np.ndarray) -> float:
    """95th-percentile symmetric Hausdorff distance, in pixels. Complements Dice:
    Dice is dominated by the interior, HD95 by the worst part of the boundary."""
    pa, pb = _boundary_points(a), _boundary_points(b)
    if pa.size == 0 or pb.size == 0:
        return float("nan")
    d_ab = np.sqrt(((pa[:, None, :] - pb[None, :, :]) ** 2).sum(-1)).min(axis=1)
    d_ba = np.sqrt(((pb[:, None, :] - pa[None, :, :]) ** 2).sum(-1)).min(axis=1)
    return float(np.percentile(np.concatenate([d_ab, d_ba]), 95))


# --------------------------------------------------------------------------
# roughness
# --------------------------------------------------------------------------

def largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    """Biggest connected component's outline. Segmenters sometimes emit specks;
    shape descriptors on a 3-pixel blob are noise, so we describe the main body
    and report `n_components` separately rather than silently merging them."""
    cnts, _ = cv2.findContours((np.asarray(mask) > 0).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = [c for c in cnts if len(c) >= 5]
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)


def radial_profile(contour: np.ndarray, n_angles: int = N_ANGLES) -> Optional[np.ndarray]:
    """r(theta) on a uniform angular grid, normalised to mean radius 1.

    Empty angular bins (possible on a non-convex contour sampled coarsely) are
    filled by circular interpolation from their neighbours, so the FFT below
    never sees a spurious step edge that would masquerade as jaggedness."""
    c = np.asarray(contour, dtype=np.float64)
    if len(c) < 8:
        return None
    centre = c.mean(axis=0)
    d = c - centre
    r = np.hypot(d[:, 0], d[:, 1])
    th = np.mod(np.arctan2(d[:, 1], d[:, 0]), 2 * math.pi)

    bins = np.minimum((th / (2 * math.pi) * n_angles).astype(int), n_angles - 1)
    # Seed with -inf, NOT NaN: np.maximum propagates NaN, so a NaN-filled array
    # would come back entirely NaN and every profile would silently be discarded.
    prof = np.full(n_angles, -np.inf)
    # Outermost crossing per bin -- the documented star-shaped approximation.
    np.maximum.at(prof, bins, r)
    known = np.isfinite(prof) & (prof > 0)
    if known.sum() < n_angles // 4:
        return None
    if not known.all():
        idx = np.arange(n_angles)
        ki = idx[known]
        # Wrap by tiling once on each side so interpolation is circular.
        prof = np.interp(idx, np.concatenate([ki - n_angles, ki, ki + n_angles]),
                         np.tile(prof[known], 3))
    m = prof.mean()
    return prof / m if m > 0 else None


def harmonic_amplitudes(profile: np.ndarray) -> np.ndarray:
    """amp[k] for the normalised radial profile: amplitude of the k-th harmonic
    as a fraction of mean radius, i.e. the same units as shape_probe's BASE."""
    n = len(profile)
    spec = np.fft.rfft(profile - profile.mean())
    return 2.0 * np.abs(spec) / n


def roughness(mask: np.ndarray) -> Dict[str, float]:
    """Shape descriptors for one binary mask.

    circularity  4*pi*A/P^2 -- 1.0 for a circle, falls as the boundary convolves.
                 Sensitive to fine jaggedness, and to rasterisation, so compare
                 it only between masks of the same size.
    solidity     A / A(convex hull) -- makes no star-shaped assumption; the one
                 that responds to concave bites (the `geographic` class).
    lobe_amp     RMS harmonic amplitude over k=3-7  (lobulation)
    jag_amp      RMS harmonic amplitude over k=8-22 (spiculation / jaggedness)
    elongation   major/minor axis of the fitted ellipse; reported so a change in
                 lobe_amp is not confused with the lesion simply being oval.
    """
    m = (np.asarray(mask) > 0).astype(np.uint8)
    n_cc = int(cv2.connectedComponents(m)[0] - 1)
    out = {"area_px": float(m.sum()), "n_components": float(n_cc),
           "circularity": float("nan"), "solidity": float("nan"),
           "lobe_amp": float("nan"), "jag_amp": float("nan"),
           "elongation": float("nan")}

    c = largest_contour(m)
    if c is None:
        return out

    area = cv2.contourArea(c.astype(np.float32))
    perim = cv2.arcLength(c.astype(np.float32), True)
    if perim > 0:
        out["circularity"] = float(4 * math.pi * area / (perim ** 2))
    hull = cv2.convexHull(c.astype(np.float32))
    hull_area = cv2.contourArea(hull)
    if hull_area > 0:
        out["solidity"] = float(area / hull_area)
    if len(c) >= 5:
        (_, (ax1, ax2), _) = cv2.fitEllipse(c.astype(np.float32))
        lo, hi = sorted((ax1, ax2))
        out["elongation"] = float(hi / lo) if lo > 0 else float("nan")

    prof = radial_profile(c)
    if prof is not None:
        amp = harmonic_amplitudes(prof)
        def band(lo: int, hi: int) -> float:
            hi = min(hi, len(amp) - 1)
            return float(np.sqrt((amp[lo:hi + 1] ** 2).sum())) if hi >= lo else float("nan")
        out["lobe_amp"] = band(*LOBE_K)
        out["jag_amp"] = band(*JAG_K)
    return out
