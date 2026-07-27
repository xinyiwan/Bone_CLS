"""
Optional QC overlay: draw the segmentation BOUNDARY (thin contour, not filled)
on the normalized image. This is for eyeballing that slice selection / cropping
is sensible -- it is NOT the image fed to the model.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def to_uint8(img2d_float01: np.ndarray) -> np.ndarray:
    return (np.clip(np.asarray(img2d_float01), 0.0, 1.0) * 255).astype(np.uint8)


def draw_contour_overlay(
    img2d_float01: np.ndarray,
    mask2d: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),  # RGB
    thickness: int = 2,
) -> np.ndarray:
    """Return an RGB uint8 image with the mask outline drawn on top."""
    rgb = cv2.cvtColor(to_uint8(img2d_float01), cv2.COLOR_GRAY2RGB)
    binary = (np.asarray(mask2d) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, contours, -1, color, thickness)
    return rgb
