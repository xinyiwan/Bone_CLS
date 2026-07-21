"""
PNG writing (8-bit) and the running metadata CSV.

PNGs are 128x128 8-bit -- for a VLM, not for quantitative reanalysis.

Metadata schema (flattened: one row per image/plane, no mixing of orientations):
    case_id, feature_name, modality, plane, slice_index, image_path,
    crop_bbox, margin_used
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from overlay import to_uint8

METADATA_FIELDS = [
    "case_id",
    "feature_name",
    "modality",
    "plane",
    "slice_index",
    "image_path",
    "crop_bbox",
    "margin_used",
]


def save_gray_png(img2d_float01: np.ndarray, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), to_uint8(img2d_float01))


def save_rgb_png(rgb_uint8: np.ndarray, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.asarray(rgb_uint8), cv2.COLOR_RGB2BGR))


class MetadataWriter:
    """Append rows to a CSV as the pipeline runs (flush per row so a crash mid-
    batch still leaves a usable partial CSV)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=METADATA_FIELDS)
        self._writer.writeheader()
        self._fh.flush()

    def write_row(self, row: Dict[str, object]) -> None:
        self._writer.writerow({k: row.get(k, "") for k in METADATA_FIELDS})
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "MetadataWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def join_field(values: List[object]) -> str:
    """Semicolon-join per-image fields (paths, slices, bboxes) for one feature."""
    return ";".join(str(v) for v in values)
