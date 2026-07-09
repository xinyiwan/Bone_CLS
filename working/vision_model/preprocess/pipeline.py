"""
Glue: turn one subject + one FeatureSpec into PNG crops + a metadata row.

Each step (load / normalize / slice / crop / resize / overlay / save) lives in
its own module and is independently testable; this file only orchestrates.

Output layout:
    {out_root}/{subject}/{feature}/{modality}_{plane}_{slice}.png
    {out_root}/{subject}/{feature}/{modality}_{plane}_{slice}_overlay.png   (if enabled)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from config import FeatureSpec
from cropping import apply_mask, crop_with_bbox, expand_bbox, margin_px_from_mm, mask_bbox_2d
from normalize import normalize_intensity
from outputs import join_field, save_gray_png, save_rgb_png
from overlay import draw_contour_overlay
from resize import resize_image, resize_mask
from slicing import PLANE_AXES, extract_slice, find_top_k_area_slices, inplane_spacing
from volume_io import load_volume_and_mask

log = logging.getLogger("preprocess")


@dataclass
class PipelineOptions:
    out_size: tuple = (128, 128)
    norm_method: str = "minmax"
    pad_mode: str = "clip"           # 'clip' or 'pad' for out-of-bounds crops
    crop_mode: str = "bbox"          # 'bbox' (keep context) or 'masked' (lesion only)
    mask_dilate_px: int = 0          # only for 'masked': keep a rim of tissue
    foreground_only: bool = True
    overlay: bool = False
    img_pattern: str = "{modality}.nii.gz"
    seg_pattern: str = "{modality}_seg.nii.gz"


class MissingInput(FileNotFoundError):
    """A required modality volume or mask is absent for a subject."""


def _resolve_paths(data_root: Path, subject: str, modality: str, opt: PipelineOptions):
    img = data_root / subject / opt.img_pattern.format(modality=modality)
    seg = data_root / subject / opt.seg_pattern.format(modality=modality)
    # tolerate .nii vs .nii.gz
    if not img.exists() and str(img).endswith(".nii.gz"):
        alt = Path(str(img)[:-3])
        img = alt if alt.exists() else img
    return img, seg


def process_feature(
    subject: str,
    spec: FeatureSpec,
    data_root: Path,
    out_root: Path,
    opt: PipelineOptions,
) -> Optional[Dict[str, object]]:
    """Process every requirement/plane/slice for one feature. Returns a metadata
    row dict, or None if nothing usable was produced. Raises MissingInput if a
    required modality/mask file is absent (batch driver logs & skips)."""
    modalities: List[str] = []
    planes: List[str] = []
    slices: List[int] = []
    image_paths: List[str] = []
    bboxes: List[str] = []
    margins: List[str] = []

    feat_dir = out_root / subject / spec.name

    for req in spec.requirements:
        vol_path, seg_path = _resolve_paths(data_root, subject, req.modality, opt)
        if not vol_path.exists() or not seg_path.exists():
            raise MissingInput(f"{subject}: missing {req.modality} vol/seg ({vol_path.name}, {seg_path.name})")

        vol, mask, spacing, _affine = load_volume_and_mask(vol_path, seg_path)
        vol_norm = normalize_intensity(vol, method=opt.norm_method, foreground_only=opt.foreground_only)

        for plane in req.planes:
            if plane not in PLANE_AXES:
                log.warning("%s/%s: unknown plane %r, skipping", subject, spec.name, plane)
                continue

            top = find_top_k_area_slices(mask, plane, spec.top_k)
            if not top:
                log.warning("%s/%s: empty mask in %s plane for %s", subject, spec.name, plane, req.modality)
                continue

            row_sp, col_sp = inplane_spacing(spacing, plane)
            for idx in top:
                img2d = extract_slice(vol_norm, plane, idx)
                m2d = extract_slice(mask, plane, idx)

                bbox = mask_bbox_2d(m2d)
                if bbox is None:  # shouldn't happen (top-k filters empty), but be safe
                    bbox = (0, img2d.shape[0] - 1, 0, img2d.shape[1] - 1)

                if spec.margin_mm is not None:
                    m_rows, m_cols = margin_px_from_mm(spec.margin_mm, row_sp, col_sp)
                    margin_desc = f"{spec.margin_mm}mm=({m_rows},{m_cols})px"
                else:
                    m_rows = m_cols = int(spec.margin_px or 0)
                    margin_desc = f"{m_rows}px"

                ebox = expand_bbox(bbox, m_rows, m_cols)
                crop_img = crop_with_bbox(img2d, ebox, mode=opt.pad_mode, pad_value=0.0)

                # 'bbox' keeps surrounding tissue; 'masked' drops everything
                # outside the segmentation (feature-level override wins).
                mode = spec.crop_mode or opt.crop_mode
                need_mask = opt.overlay or mode == "masked"
                crop_m = crop_with_bbox(m2d, ebox, mode=opt.pad_mode, pad_value=0) if need_mask else None
                if mode == "masked":
                    crop_img = apply_mask(crop_img, crop_m, background=0.0, dilate_px=opt.mask_dilate_px)

                out_img = resize_image(crop_img, opt.out_size)

                png = feat_dir / f"{req.modality}_{plane}_{idx}.png"
                save_gray_png(out_img, png)

                if opt.overlay:
                    out_mask = resize_mask(crop_m, opt.out_size)
                    save_rgb_png(draw_contour_overlay(out_img, out_mask),
                                 feat_dir / f"{req.modality}_{plane}_{idx}_overlay.png")

                modalities.append(req.modality)
                planes.append(plane)
                slices.append(idx)
                image_paths.append(str(png))
                bboxes.append(f"[{ebox[0]},{ebox[1]},{ebox[2]},{ebox[3]}]")
                margins.append(f"{margin_desc}|{mode}")

    if not image_paths:
        return None

    return {
        "case_id": subject,
        "feature_name": spec.name,
        "modality": join_field(sorted(set(modalities))),
        "plane": join_field(planes),
        "slice_indices": join_field(slices),
        "image_paths": join_field(image_paths),
        "crop_bbox": join_field(bboxes),
        "margin_used": join_field(margins),
    }


def process_subject(
    subject: str,
    specs: List[FeatureSpec],
    data_root: Path,
    out_root: Path,
    opt: PipelineOptions,
    meta_writer,
) -> None:
    """Run all features for one subject. Errors in one feature are logged and
    skipped so the batch never dies on a single bad subject/feature."""
    for spec in specs:
        try:
            row = process_feature(subject, spec, data_root, out_root, opt)
            if row is not None:
                meta_writer.write_row(row)
                log.info("%s / %s: %d image(s)", subject, spec.name, row["image_paths"].count(";") + 1)
        except MissingInput as e:
            log.warning("SKIP %s", e)
        except Exception as e:  # noqa: BLE001 -- isolate one bad feature
            log.exception("ERROR %s / %s: %s", subject, spec.name, e)
