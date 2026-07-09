"""
Resize 2D crops to the model input size.

Image: bilinear (default) -- appropriate for continuous intensities.
Mask:  nearest-neighbour  -- it is a label map; never interpolate labels.

cv2.resize takes dsize as (width, height); we pass (W, H) accordingly.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def resize_image(img2d: np.ndarray, size: Tuple[int, int] = (128, 128), interp: int = cv2.INTER_LINEAR) -> np.ndarray:
    """Resize a 2D intensity image to (H, W). Use cv2.INTER_CUBIC for sharper."""
    h, w = size
    return cv2.resize(np.asarray(img2d, dtype=np.float32), (w, h), interpolation=interp)


def resize_mask(mask2d: np.ndarray, size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """Resize a 2D label map with nearest-neighbour (labels preserved)."""
    h, w = size
    out = cv2.resize(np.asarray(mask2d).astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST)
    return out.astype(mask2d.dtype if hasattr(mask2d, "dtype") else np.int16)
