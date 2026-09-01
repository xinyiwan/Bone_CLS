"""
Glue: turn one case + one FeatureSpec into PNG crops + a metadata row.

Each step (load / normalize / slice / crop / resize / overlay / save) lives in
its own module and is independently testable; this file only orchestrates.

Which files a (modality, plane) requirement maps to is delegated to a RESOLVER
callable (run.py builds one from pairs.find_pairs + the classified-sequence
table). A resolver takes (case_id, modality, plane) and returns (vol_path,
seg_path), or None when that combination isn't available for the case. Keeping
resolution behind this seam is what makes the slice/crop/normalize steps unit-
testable without any real data.

Output layout:
    {out_root}/{case_id}/{feature}/{modality}_{plane}_{slice}.png
    {out_root}/{case_id}/{feature}/{modality}_{plane}_{slice}_overlay.png   (if enabled)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from config import FeatureSpec
from cropping import (apply_mask, crop_with_bbox, expand_bbox, margin_px_from_mm,
                      mask_bbox_2d, union_bbox)
from normalize import normalize_intensity
from outputs import join_field, save_gray_png, save_rgb_png
from overlay import draw_contour_overlay
from resize import resize_image, resize_mask
from slicing import (PLANE_AXES, extract_slice, find_top_k_area_slices,
                     find_tumor_slices, inplane_spacing)
from volume_io import load_volume_and_mask

log = logging.getLogger("preprocess")

# (case_id, modality, plane) -> (volume_path, seg_path) or None if unavailable.
Resolver = Callable[[str, str, str], Optional[Tuple[Path, Path]]]


@dataclass
class PipelineOptions:
    out_size: tuple = (128, 128)
    norm_method: str = "minmax"
    pad_mode: str = "clip"           # 'clip' or 'pad' for out-of-bounds crops
    crop_mode: str = "bbox"          # 'bbox' (keep context) or 'masked' (lesion only)
    mask_dilate_px: int = 0          # only for 'masked': keep a rim of tissue
    foreground_only: bool = True
    overlay: bool = False
    # Write the binary mask crop alongside each image (see seg_probe/): needed to
    # score an automatic segmenter against the radiologist in the crop's own frame.
    save_mask: bool = False


def _safe_name(s: str) -> str:
    """Filesystem-safe token for filenames (sequence labels are already clean,
    but a case_id may contain '/')."""
    return str(s).replace("/", "__").replace(" ", "_")


def process_feature(
    case_id: str,
    spec: FeatureSpec,
    out_root: Path,
    opt: PipelineOptions,
    resolve: Resolver,
    vol_cache: Optional[Dict[str, tuple]] = None,
    ground_truth: str = "unknown",
) -> Optional[List[Dict[str, object]]]:
    """Process every requirement/plane/slice for one feature. Returns a list of
    metadata row dicts, one per (plane, slice) pair. Each row is independent:
    different planes/orientations are NOT grouped together. Returns None (or empty
    list) if nothing usable was produced. Missing (modality, plane) combinations
    are logged and skipped, not fatal.

    `ground_truth` is stamped onto every row for this feature (it's a per-subject,
    per-feature label, independent of plane/slice); "unknown" if unavailable."""
    if vol_cache is None:
        vol_cache = {}
    rows: List[Dict[str, object]] = []

    feat_dir = out_root / _safe_name(case_id) / spec.name

    for req in spec.requirements:
        for plane in req.planes:
            if plane not in PLANE_AXES:
                log.warning("%s/%s: unknown plane %r, skipping", case_id, spec.name, plane)
                continue

            hit = resolve(case_id, req.modality, plane)
            if hit is None:
                log.warning("%s/%s: no %s scan in %s plane, skipping", case_id, spec.name, req.modality, plane)
                continue
            vol_path, seg_path = hit

            # Reslicing the same file for several planes / features -> load once.
            key = str(vol_path)
            if key not in vol_cache:
                vol, mask, spacing, _affine = load_volume_and_mask(vol_path, seg_path)
                vol_norm = normalize_intensity(vol, method=opt.norm_method, foreground_only=opt.foreground_only)
                vol_cache[key] = (vol_norm, mask, spacing)
            vol_norm, mask, spacing = vol_cache[key]

            stacked = spec.slice_mode == "stack"
            if stacked:
                top = find_tumor_slices(mask, plane, max_slices=spec.max_slices)
            else:
                top = find_top_k_area_slices(mask, plane, spec.top_k)
            if not top:
                log.warning("%s/%s: empty mask in %s plane for %s", case_id, spec.name, plane, req.modality)
                continue

            # STACK: one box for the whole run of slices, computed BEFORE the
            # loop and reused for every one of them. Cropping each slice to its
            # own box and resizing them all to out_size would rescale each slice
            # independently, so the lesion would occupy a similar fraction of
            # every frame regardless of its true extent -- erasing the taper and
            # the drift that are the only reasons to look at a stack at all.
            shared_bbox = None
            if stacked:
                shared_bbox = union_bbox(mask_bbox_2d(extract_slice(mask, plane, i)) for i in top)
                log.info("%s/%s: stack of %d %s slice(s) %d-%d, shared bbox %s",
                         case_id, spec.name, len(top), plane, top[0], top[-1], shared_bbox)

            row_sp, col_sp = inplane_spacing(spacing, plane)
            for stack_pos, idx in enumerate(top):
                img2d = extract_slice(vol_norm, plane, idx)
                m2d = extract_slice(mask, plane, idx)

                bbox = shared_bbox if stacked else mask_bbox_2d(m2d)
                # Empty is impossible for top-k (it filters them) but NOT for a
                # stack: fill_gaps keeps interior slices whose mask is empty, and
                # those must still be cropped to the shared box, not to the whole
                # image -- otherwise a gap slice silently breaks the registration.
                if bbox is None:
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
                need_mask = opt.overlay or opt.save_mask or mode == "masked"
                crop_m = crop_with_bbox(m2d, ebox, mode=opt.pad_mode, pad_value=0) if need_mask else None
                if mode == "masked":
                    crop_img = apply_mask(crop_img, crop_m, background=0.0, dilate_px=opt.mask_dilate_px)

                out_img = resize_image(crop_img, opt.out_size)

                png = feat_dir / f"{req.modality}_{plane}_{idx}.png"
                save_gray_png(out_img, png)

                out_mask = resize_mask(crop_m, opt.out_size) if need_mask else None

                if opt.overlay:
                    save_rgb_png(draw_contour_overlay(out_img, out_mask),
                                 feat_dir / f"{req.modality}_{plane}_{idx}_overlay.png")

                # The mask crop, saved as a plain binary PNG in the SAME geometry
                # as image_path. Without it any downstream comparison against an
                # automatic segmenter (seg_probe/) has to re-derive the crop from
                # the NIfTI and re-do the resize, which is where off-by-one and
                # interpolation mismatches creep in and silently deflate Dice.
                mask_png = ""
                if opt.save_mask:
                    mask_png = feat_dir / f"{req.modality}_{plane}_{idx}_mask.png"
                    save_gray_png((out_mask > 0).astype(np.float32), mask_png)
                    mask_png = str(mask_png)

                # One row per image (plane/slice).
                rows.append({
                    "case_id": case_id,
                    "feature_name": spec.name,
                    "modality": req.modality,
                    "plane": plane,
                    "slice_index": idx,
                    "image_path": str(png),
                    "mask_path": mask_png,
                    "crop_bbox": f"[{ebox[0]},{ebox[1]},{ebox[2]},{ebox[3]}]",
                    "margin_used": f"{margin_desc}|{mode}",
                    # Stack bookkeeping. slice_index alone is NOT enough to
                    # reassemble a stack: a loader sorting on it cannot tell a
                    # complete run from one missing its middle, and cannot know
                    # when to stop. stack_pos/stack_size make the run explicit
                    # and self-checking. Blank in top_k mode, where the slices
                    # are independent samples and there is no run to rebuild.
                    "slice_mode": spec.slice_mode,
                    "stack_pos": stack_pos if stacked else "",
                    "stack_size": len(top) if stacked else "",
                    "ground_truth_label": ground_truth,
                })

    return rows if rows else None


def process_case(
    case_id: str,
    specs: List[FeatureSpec],
    out_root: Path,
    opt: PipelineOptions,
    resolve: Resolver,
    meta_writer,
    gt_for: Optional[Callable[[str, FeatureSpec], str]] = None,
) -> None:
    """Run all features for one case. Errors in one feature are logged and
    skipped so the batch never dies on a single bad case/feature. The volume
    cache is shared across features so a scan needed by several features (or in
    several planes) is loaded/normalized only once.

    `gt_for(case_id, spec) -> label` supplies the ground-truth label per feature
    (from the assessment JSON); when None, every row is labelled "unknown"."""
    vol_cache: Dict[str, tuple] = {}
    for spec in specs:
        try:
            ground_truth = gt_for(case_id, spec) if gt_for else "unknown"
            rows = process_feature(case_id, spec, out_root, opt, resolve, vol_cache,
                                   ground_truth=ground_truth)
            if rows:
                for row in rows:
                    meta_writer.write_row(row)
                log.info("%s / %s: %d image(s) [gt=%s]", case_id, spec.name, len(rows), ground_truth)
        except Exception as e:  # noqa: BLE001 -- isolate one bad feature
            log.exception("ERROR %s / %s: %s", case_id, spec.name, e)
