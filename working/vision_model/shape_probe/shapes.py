"""
Synthetic shape rasterizers for the perception probe.

The whole point of the probe is that the drawn shape is the ONLY thing that
distinguishes classes, so every shape is drawn the same way the real
segmentation contour is drawn in `preprocess/overlay.py`:

    RGB uint8 image, red (255, 0, 0) polyline, thickness 2, anti-aliased off

so the model sees an overlay of the same "visual species" as the radiologist
contour it gets in the real run. If you change the real overlay's colour or
thickness, change `DEFAULT_COLOR` / `DEFAULT_THICKNESS` here too -- otherwise
the probe stops being a proxy for the real task.

All shapes are inscribed in a circle of radius `radius_px` around `center`, so
the four classes cover comparable image area and cannot be told apart by size.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import cv2
import numpy as np

SHAPES = ("circle", "square", "triangle", "star")

DEFAULT_COLOR = (255, 0, 0)  # RGB, matches preprocess.overlay.draw_contour_overlay
DEFAULT_THICKNESS = 2


def _polygon(center: Tuple[float, float], radius: float, n: int, rotation_deg: float) -> np.ndarray:
    """`n` vertices evenly spaced on the circumscribed circle."""
    cx, cy = center
    phi = math.radians(rotation_deg) - math.pi / 2  # -90deg -> first vertex points up
    pts = [
        (cx + radius * math.cos(phi + 2 * math.pi * i / n),
         cy + radius * math.sin(phi + 2 * math.pi * i / n))
        for i in range(n)
    ]
    return np.array(pts, dtype=np.int32)


def _star(center: Tuple[float, float], radius: float, rotation_deg: float,
          points: int = 5, inner_ratio: float = 0.4) -> np.ndarray:
    """A `points`-pointed star: outer/inner vertices alternating."""
    cx, cy = center
    phi = math.radians(rotation_deg) - math.pi / 2
    pts = []
    for i in range(2 * points):
        r = radius if i % 2 == 0 else radius * inner_ratio
        a = phi + math.pi * i / points
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return np.array(pts, dtype=np.int32)


def shape_polygon(shape: str, center: Tuple[float, float], radius: float,
                  rotation_deg: float = 0.0) -> np.ndarray | None:
    """Vertices for `shape`, or None for 'circle' (drawn analytically, not as a
    polygon, so it stays smooth at small radii)."""
    if shape == "circle":
        return None
    if shape == "square":
        return _polygon(center, radius, 4, rotation_deg + 45.0)  # +45 -> flat sides at rotation=0
    if shape == "triangle":
        return _polygon(center, radius, 3, rotation_deg)
    if shape == "star":
        return _star(center, radius, rotation_deg)
    raise ValueError(f"Unknown shape {shape!r} (expected one of {SHAPES})")


def draw_shape(
    rgb: np.ndarray,
    shape: str,
    center: Tuple[float, float],
    radius: float,
    rotation_deg: float = 0.0,
    color: Sequence[int] = DEFAULT_COLOR,
    thickness: int = DEFAULT_THICKNESS,
    filled: bool = False,
) -> np.ndarray:
    """Draw `shape` onto a copy of an RGB uint8 image and return it."""
    out = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8).copy())
    col = tuple(int(c) for c in color)
    t = -1 if filled else int(thickness)

    if shape == "circle":
        cv2.circle(out, (int(round(center[0])), int(round(center[1]))), int(round(radius)), col, t)
    else:
        # generate a randon rotation degree
        rotation_deg = np.random.uniform(0, 360)
        poly = shape_polygon(shape, center, radius, rotation_deg)
        if filled:
            cv2.fillPoly(out, [poly], col)
        else:
            cv2.polylines(out, [poly], isClosed=True, color=col, thickness=int(thickness))
    return out
