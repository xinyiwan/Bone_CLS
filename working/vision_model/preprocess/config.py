"""
Feature -> modality -> plane(s) -> margin mapping.

The mapping is DATA, not code: which features need which modality/plane/margin
is read from a YAML or CSV config so features can be added over time without
touching the pipeline. See feature_config.yaml for the canonical format.

A feature may need several images (e.g. "shape" = T1 axial + coronal). We model
that as a list of Requirement(modality, planes); the pipeline emits one output
image per (requirement, plane, selected-slice) and groups them into a single
metadata row per (subject, feature).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Requirement:
    """One modality rendered in one or more planes."""
    modality: str
    planes: List[str]


@dataclass
class FeatureSpec:
    name: str
    requirements: List[Requirement] = field(default_factory=list)
    # Margin can be given in mm (converted per-axis with voxel spacing) or in
    # pixels. mm takes precedence when both are set; that's the recommended one
    # for anisotropic MRI.
    margin_mm: Optional[float] = None
    margin_px: Optional[int] = None
    top_k: int = 1  # slices per plane, ranked by tumour cross-sectional area
    # 'top_k'  -> the `top_k` largest-area slices, each cropped to its OWN bbox.
    #             One row per slice; the slices are independent samples.
    # 'stack'  -> every tumour-bearing slice, in anatomical order, all cropped to
    #             the SAME (union) bbox so they share a frame and can be read as a
    #             volume. `top_k` is ignored; `max_slices` caps the count.
    slice_mode: str = "top_k"
    max_slices: int = 85  # stack mode only; MedGemma 1.5's per-query ceiling
    # How the rectangle's pixels are treated:
    #   'bbox'   -> keep everything in the box (lesion + surrounding tissue)
    #   'masked' -> zero out pixels outside the segmentation (lesion only)
    # None -> fall back to the pipeline/CLI default. Set per-feature since e.g.
    # 'enhancement' wants context but 'fluid_fluid_level' may not.
    crop_mode: Optional[str] = None
    # Key of this feature under the assessment JSON's "imaging_features" block
    # (ground-truth source), when it differs from `name`. e.g. shape ->
    # tumor_shape. None -> use `name` as the key.
    assessment_key: Optional[str] = None


def load_feature_config(path: str | Path) -> List[FeatureSpec]:
    path = Path(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        return _load_yaml(path)
    if path.suffix.lower() == ".csv":
        return _load_csv(path)
    raise ValueError(f"Unsupported config type: {path.suffix} (use .yaml or .csv)")


def _load_yaml(path: Path) -> List[FeatureSpec]:
    import yaml  # local import so CSV-only users don't need PyYAML

    with open(path) as fh:
        raw = yaml.safe_load(fh)

    specs: List[FeatureSpec] = []
    for feat in raw["features"]:
        reqs = [
            Requirement(modality=r["modality"], planes=list(r["planes"]))
            for r in feat["requirements"]
        ]
        specs.append(
            FeatureSpec(
                name=feat["name"],
                requirements=reqs,
                margin_mm=feat.get("crop_margin_mm"),
                margin_px=feat.get("crop_margin_px"),
                top_k=int(feat.get("top_k", 1)),
                slice_mode=str(feat.get("slice_mode", "top_k")).strip(),
                max_slices=int(feat.get("max_slices", 85)),
                crop_mode=feat.get("crop_mode"),
                assessment_key=feat.get("assessment_key"),
            )
        )
    for s in specs:
        if s.slice_mode not in ("top_k", "stack"):
            raise ValueError(f"feature {s.name!r}: slice_mode must be 'top_k' or 'stack', "
                             f"got {s.slice_mode!r}")
    return specs


def _load_csv(path: Path) -> List[FeatureSpec]:
    """
    Flat CSV, one row per (feature, modality). Columns:
        feature_name, modality, plane, crop_margin_mm[, top_k]
    `plane` may hold several planes joined by ';' or '|' (e.g. "axial;coronal").
    Rows sharing feature_name are grouped into one FeatureSpec.
    """
    by_name: dict[str, FeatureSpec] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            name = row["feature_name"].strip()
            planes = [p.strip() for p in row["plane"].replace("|", ";").split(";") if p.strip()]
            spec = by_name.setdefault(name, FeatureSpec(name=name))
            spec.requirements.append(Requirement(modality=row["modality"].strip(), planes=planes))
            if row.get("crop_margin_mm"):
                spec.margin_mm = float(row["crop_margin_mm"])
            if row.get("top_k"):
                spec.top_k = int(row["top_k"])
            if row.get("crop_mode"):
                spec.crop_mode = row["crop_mode"].strip()
            if row.get("assessment_key"):
                spec.assessment_key = row["assessment_key"].strip()
    return list(by_name.values())


def load_sequence_aliases(path: str | Path) -> dict:
    """Optional {config_modality -> classified_label} map, read from a top-level
    `sequence_aliases:` block in a YAML config. Lets the feature config use short
    names (T1C) that map onto your classifier's labels (T1W_nFS_CE). Returns {}
    for CSV configs or when the block is absent."""
    path = Path(path)
    if path.suffix.lower() not in {".yaml", ".yml"}:
        return {}
    import yaml

    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    return dict(raw.get("sequence_aliases", {}))
