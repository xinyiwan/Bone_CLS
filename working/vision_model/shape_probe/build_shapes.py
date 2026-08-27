"""
Build the pseudo-segmentation dataset for the perception probe.

QUESTION THIS ANSWERS
    In the real run we hand MedGemma a crop with the radiologist's red tumour
    contour burned in (`run_medgemma.py --use-contour`) and assume it can see
    that contour. Can it? This script replaces the true contour with a
    *synthetic* one drawn at the same place, in the same colour and thickness,
    and asks only which shape it is.

    Two ladders, selected with --shape-set:

    icons     circle / square / triangle / star. Tests PERCEPTION -- is the
              overlay visible at all. Each class has a distinct vertex count,
              so it is solvable by corner-counting; near-perfect accuracy here
              means the overlay is legible, not that margin shape is legible.

    clinical  the five margin classes of the `shape` feature in
              medgemma_pilot/feature_prompts.yaml, all generated from ONE radial
              equation with different parameters (see shapes.py). Tests
              DISCRIMINATION -- can it tell 5 smooth lobes from 20 jagged spikes
              when both are "bumpy". --difficulty sweeps deformation amplitude,
              so the output is a psychometric curve rather than one number, and
              it is an UPPER BOUND on the real task: same question, same
              vocabulary, same prompt path, but perfect labels and no anatomy.

DESIGN
    - Input is the metadata CSV the preprocess pipeline already wrote. We do NOT
      re-read NIfTI volumes and we do NOT touch the preprocess code: the crops
      are already centred on the lesion bbox (bbox + margin), so the lesion
      centre IS the crop centre, and the lesion extent is recoverable from the
      `crop_bbox` / `margin_used` columns.
    - Shape radius is derived from the true lesion extent, so the shape covers
      roughly the area the real contour would -- the probe stays at the real
      task's difficulty rather than being an easy big-shape-on-blank test.
    - Shape assignment is balanced (round-robin over a seeded shuffle), so a
      model that always answers "circle" scores at chance, not above it.
    - Rotation is randomised per image so orientation can't be memorised;
      'circle' ignores it.

CAVEAT (small, worth knowing)
    `crop_bbox` records the *requested* box. With the pipeline's default
    `--pad-mode clip`, lesions touching the image border get a smaller actual
    crop, so for those rows the shape centre is off by a few pixels and the
    radius slightly overestimates. It does not affect the probe's validity (the
    shape is still fully drawn and centred on tissue); use `--pad-mode pad` in
    preprocess if you want it exact.

OUTPUT
    {out_root}/{case_id}/{feature}/{stem}_shape-{shape}.png
    {out_root}/shape_metadata.csv   (schema below, in SHAPE_FIELDS)
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import re
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from shapes import (  # noqa: E402
    DIFFICULTY_PRESETS,
    SHAPE_SETS,
    clinical_polygon,
    draw_poly,
    draw_shape,
)

log = logging.getLogger("shape_probe.build")

SHAPE_FIELDS = [
    "case_id",
    "feature_name",
    "modality",
    "plane",
    "slice_index",
    "source_image_path",   # the plain crop this was drawn on
    "image_path",          # the shape image -- this is what the model sees
    "shape",               # GROUND TRUTH for the probe
    "shape_set",           # icons | clinical
    "difficulty",          # deformation amplitude multiplier (clinical set only)
    "shape_params",        # per-image generator params, e.g. "lobe_k=5;lobe_amp=0.24"
    "background",          # mri | blank | noise
    "filled",
    "center_xy",
    "radius_px",
    "rotation_deg",
    "lesion_frac",         # lesion extent / crop extent, before resize
]

BBOX_RE = re.compile(r"-?\d+")
# "12.0mm=(5,7)px|bbox"  or  "5px|bbox"
MM_MARGIN_RE = re.compile(r"=\((\d+),\s*(\d+)\)px")
PX_MARGIN_RE = re.compile(r"^(\d+)px")


def parse_bbox(s: str) -> Optional[Tuple[int, int, int, int]]:
    nums = [int(n) for n in BBOX_RE.findall(str(s))]
    return tuple(nums[:4]) if len(nums) >= 4 else None  # type: ignore[return-value]


def parse_margin_px(s: str) -> Tuple[int, int]:
    """(margin_rows, margin_cols) in pixels from the `margin_used` column."""
    head = str(s).split("|", 1)[0].strip()
    m = MM_MARGIN_RE.search(head)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = PX_MARGIN_RE.match(head)
    if m:
        return int(m.group(1)), int(m.group(1))
    return 0, 0


def lesion_fraction(row) -> Tuple[float, float]:
    """Fraction of the crop's (height, width) occupied by the lesion bbox.
    Falls back to (0.5, 0.5) when the metadata columns are missing/unparseable."""
    bbox = parse_bbox(row.get("crop_bbox", ""))
    if bbox is None:
        return 0.5, 0.5
    r0, r1, c0, c1 = bbox
    m_rows, m_cols = parse_margin_px(row.get("margin_used", ""))
    ext_r, ext_c = (r1 - r0 + 1), (c1 - c0 + 1)
    if ext_r <= 0 or ext_c <= 0:
        return 0.5, 0.5
    fr = max(0.05, min(1.0, (ext_r - 2 * m_rows) / ext_r))
    fc = max(0.05, min(1.0, (ext_c - 2 * m_cols) / ext_c))
    return fr, fc


def make_background(src_path: Path, mode: str, size: Tuple[int, int], rng: random.Random) -> np.ndarray:
    """RGB uint8 canvas. 'mri' = the real crop (the condition we care about);
    'blank'/'noise' are controls that isolate "can it see shapes at all" from
    "can it see shapes over anatomy".

    `rng` MUST be a stream dedicated to backgrounds, never the one that draws
    shapes. Only 'noise' consumes from it, so sharing would advance the sequence
    in the noise build and not in the others -- every rotation and clinical
    parameter would then differ between the three arms, and a per-image
    difference could no longer be attributed to the background alone."""
    if mode == "mri":
        img = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(src_path)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    h, w = size
    if mode == "blank":
        return np.full((h, w, 3), 128, dtype=np.uint8)
    if mode == "noise":
        state = np.random.RandomState(rng.randrange(2**31))
        return cv2.cvtColor(state.randint(0, 256, (h, w), dtype=np.uint8), cv2.COLOR_GRAY2RGB)
    raise ValueError(f"Unknown background {mode!r} (mri|blank|noise)")


def resolve_difficulties(spec: str) -> list:
    """'hard' -> [0.35]; '1.0,0.6,0.35' -> [1.0, 0.6, 0.35]. A list makes the
    build a SWEEP: every source row is emitted once per level, so eval can plot
    accuracy against deformation amplitude instead of reporting one number."""
    out = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok in DIFFICULTY_PRESETS:
            out.append(DIFFICULTY_PRESETS[tok])
        else:
            try:
                out.append(float(tok))
            except ValueError:
                raise SystemExit(
                    f"--difficulty {tok!r} is neither a float nor one of "
                    f"{sorted(DIFFICULTY_PRESETS)}"
                )
    if not out:
        raise SystemExit("--difficulty must name at least one level")
    return out


def build(
    metadata: Path,
    out_root: Path,
    background: str = "mri",
    shape_set: str = "icons",
    difficulty: str = "easy",
    all_shapes: bool = False,
    shape_scale: float = 1.0,
    filled: bool = False,
    thickness: int = 2,
    limit: Optional[int] = None,
    seed: int = 0,
    fallback_size: Tuple[int, int] = (128, 128),
) -> Path:
    df = pd.read_csv(metadata)
    if limit:
        df = df.head(limit)
    # Two independent streams. `rng` draws everything about the SHAPE (class
    # assignment, rotation, clinical parameters); `bg_rng` draws only the noise
    # canvas. Keeping them separate is what makes the mri / blank / noise builds
    # paired: with one shared stream the noise arm's extra draw per image would
    # desynchronise it from the other two, so the same source row would carry a
    # differently-rotated shape in each arm.
    rng = random.Random(seed)
    bg_rng = random.Random(seed + 1_000_003)

    shape_names = SHAPE_SETS[shape_set]
    levels = resolve_difficulties(difficulty) if shape_set == "clinical" else [1.0]

    # Balanced assignment: a shuffled round-robin, so counts are equal +-1 and
    # the order is not correlated with case/feature.
    order = list(shape_names)
    rng.shuffle(order)
    assigned = [order[i % len(order)] for i in range(len(df))]

    out_root.mkdir(parents=True, exist_ok=True)
    out_csv = out_root / "shape_metadata.csv"
    n_written = n_failed = 0

    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SHAPE_FIELDS)
        writer.writeheader()

        for i, (_, row) in enumerate(df.iterrows()):
            src = Path(str(row["image_path"]))
            shapes_here = list(shape_names) if all_shapes else [assigned[i]]
            variants = [(s, lv) for s in shapes_here for lv in levels]
            try:
                for shape, level in variants:
                    canvas = make_background(src, background, fallback_size, bg_rng)
                    h, w = canvas.shape[:2]
                    fr, fc = lesion_fraction(row)
                    radius = 0.5 * min(h * fr, w * fc) * shape_scale
                    radius = max(4.0, min(radius, 0.48 * min(h, w)))
                    rot = rng.uniform(0, 360)
                    center = (w / 2.0, h / 2.0)

                    if shape_set == "clinical":
                        poly, params = clinical_polygon(shape, center, radius, rot,
                                                        difficulty=level, rng=rng)
                        img = draw_poly(canvas, poly, thickness=thickness, filled=filled)
                        suffix = f"_shape-{shape}_d{level:g}"
                    else:
                        params = {}
                        img = draw_shape(canvas, shape, center, radius, rot,
                                         thickness=thickness, filled=filled)
                        suffix = f"_shape-{shape}"

                    dst = (out_root / str(row["case_id"]).replace("/", "__")
                           / str(row["feature_name"]) / f"{src.stem}{suffix}.png")
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(dst), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                    writer.writerow({
                        "case_id": row.get("case_id", ""),
                        "feature_name": row.get("feature_name", ""),
                        "modality": row.get("modality", ""),
                        "plane": row.get("plane", ""),
                        "slice_index": row.get("slice_index", ""),
                        "source_image_path": str(src),
                        "image_path": str(dst),
                        "shape": shape,
                        "shape_set": shape_set,
                        "difficulty": f"{level:g}" if shape_set == "clinical" else "",
                        "shape_params": ";".join(f"{k}={v}" for k, v in params.items()),
                        "background": background,
                        "filled": filled,
                        "center_xy": f"({center[0]:.1f},{center[1]:.1f})",
                        "radius_px": f"{radius:.1f}",
                        "rotation_deg": f"{rot:.1f}",
                        "lesion_frac": f"({fr:.2f},{fc:.2f})",
                    })
                    n_written += 1
            except Exception as e:  # noqa: BLE001 -- one bad source image must not kill the batch
                n_failed += 1
                log.warning("skip %s: %s", src, e)

    log.info("wrote %d shape images (%d source rows skipped) -> %s", n_written, n_failed, out_csv)
    return out_csv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", required=True, type=Path,
                    help="preprocess metadata.csv (one row per crop)")
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--background", default="mri", choices=["mri", "blank", "noise"],
                    help="mri = real crop (main condition); blank/noise = controls")
    ap.add_argument("--shape-set", default="icons", choices=sorted(SHAPE_SETS),
                    help="icons = circle/square/triangle/star (vertex-countable, chance 25%%); "
                         "clinical = the 5 feature_prompts.yaml margin classes generated from one "
                         "radial equation (chance 20%%)")
    ap.add_argument("--difficulty", default="easy",
                    help="clinical set only: deformation amplitude, a preset "
                         f"({'/'.join(DIFFICULTY_PRESETS)}) or a float. Comma-separate for a SWEEP "
                         "(e.g. '1.0,0.6,0.35') -- each source row is emitted once per level and "
                         "eval breaks accuracy down by level.")
    ap.add_argument("--all-shapes", action="store_true",
                    help="emit every shape in the set per source image (paired design) instead of 1")
    ap.add_argument("--shape-scale", type=float, default=1.0,
                    help="multiply the lesion-derived radius (e.g. 1.5 for an easier probe)")
    ap.add_argument("--filled", action="store_true", help="solid shape instead of an outline")
    ap.add_argument("--thickness", type=int, default=2,
                    help="outline thickness; keep equal to the real contour's (2)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N source rows")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build(args.metadata, args.out_root, background=args.background, shape_set=args.shape_set,
          difficulty=args.difficulty, all_shapes=args.all_shapes, shape_scale=args.shape_scale,
          filled=args.filled, thickness=args.thickness, limit=args.limit, seed=args.seed)


if __name__ == "__main__":
    main()
