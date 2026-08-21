"""
Can an automatic segmenter replace the radiologist contour without destroying the
margin features the pipeline extracts?

WHAT THIS ANSWERS
    The medgemma pilot feeds the VLM a crop with the radiologist's contour burned
    in. Replacing that with a foundation model's segmentation would make the
    pipeline deployable (no expert 3D mask at inference) and reproducible (no
    inter-rater variance). The risk is specific: every SAM descendant has a
    strong smoothness prior, so it can match the lesion's EXTENT (high Dice)
    while erasing the lobulation and spiculation that the `shape` and margin
    features are supposed to measure. Dice alone will not show you that.

    So this scores both, per lesion:
        overlap    dice, iou, hd95          -- is it the same region?
        roughness  circularity, solidity,   -- is it the same KIND of boundary?
                   lobe_amp, jag_amp
    and reports the roughness DELTA (auto - radiologist), which is the number
    that decides whether this direction helps or quietly guts Step 3.

    lobe_amp / jag_amp are in the same units as shape_probe/shapes.py's
    deformation amplitude, so the radiologist column also tells you where real
    lesions sit on the difficulty axis that probe sweeps.

    # 1. run a segmenter (needs metadata.csv built with --save-mask)
    python run_seg_probe.py --mode segment --backend medsam \
        --metadata /results/preprocess/overlay_128/metadata.csv \
        --out /results/seg_probe/medsam/seg_results.csv

    # 2. report; pass several CSVs to compare backends in one table
    python run_seg_probe.py --mode eval --results /results/seg_probe/*/seg_results.csv

    # 3. eyeball it -- green = radiologist, red = auto, yellow = jittered prompt
    python run_seg_probe.py --mode preview \
        --results /results/seg_probe/medsam/seg_results.csv --out sheet.png

START WITH THE OFFLINE BACKENDS. `--backend box_fill` gives the Dice floor and
`--backend smooth_oracle` verifies the roughness metrics actually fire on a
smoothed boundary. Neither needs weights or a GPU. If the report does not flag
smooth_oracle, it will not flag SAM either.
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd

import backends
from geometry import box_iou, dice, hd95, iou, jitter_box, roughness, tight_box

log = logging.getLogger("seg_probe")

ROUGHNESS_KEYS = ["circularity", "solidity", "lobe_amp", "jag_amp", "elongation",
                  "area_px", "n_components"]

RESULT_FIELDS = (
    ["case_id", "feature_name", "modality", "plane", "slice_index",
     "image_path", "mask_path", "pred_mask_path", "backend", "model_id",
     "gt_box", "prompt_box", "prompt_box_iou", "jitter_shift", "jitter_scale",
     "dice", "iou", "hd95"]
    + [f"gt_{k}" for k in ROUGHNESS_KEYS]
    + [f"pred_{k}" for k in ROUGHNESS_KEYS]
)


def _load_pair(image_path: str, mask_path: str):
    """RGB image + binary GT mask, checked for identical geometry. The mask is
    written by preprocess --save-mask in the crop's own frame, so a size mismatch
    means the two came from different runs -- silently resizing here would
    deflate Dice in a way that looks like a model failure."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(image_path)
    gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise FileNotFoundError(mask_path)
    if gt.shape != img.shape:
        raise ValueError(f"mask {gt.shape} != image {img.shape} for {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), gt > 0


def segment(
    metadata: Path,
    out_path: Path,
    backend: str,
    model_id: str = "",
    shift_frac: float = 0.10,
    scale_lo: float = 0.90,
    scale_hi: float = 1.20,
    save_masks: bool = True,
    limit: Optional[int] = None,
    seed: int = 0,
    smooth_px: int = 5,
) -> None:
    df = pd.read_csv(metadata, dtype=str).fillna("")
    if "mask_path" not in df.columns:
        raise SystemExit(
            f"{metadata} has no mask_path column.\n"
            "Re-run preprocess with --save-mask: the probe scores against the mask crop in the "
            "image's own frame, and re-deriving it from the NIfTI here would introduce "
            "resize/rounding mismatches that look like segmentation error."
        )
    df = df[df["mask_path"].str.strip() != ""]
    if df.empty:
        raise SystemExit("no rows with a mask_path -- was preprocess run with --save-mask?")
    if limit:
        df = df.head(limit)

    kwargs = dict(smooth_px=smooth_px)
    if model_id:
        kwargs["model_id"] = model_id
    seg_fn = backends.make_segmenter(backend, **kwargs)
    rng = random.Random(seed)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mask_dir = out_path.parent / "pred_masks"
    n_ok = n_skip = 0

    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_FIELDS)
        writer.writeheader()

        for _, row in df.iterrows():
            try:
                img, gt = _load_pair(row["image_path"], row["mask_path"])
                gt_box = tight_box(gt)
                if gt_box is None:
                    raise ValueError("empty ground-truth mask")

                pbox = jitter_box(gt_box, gt.shape, rng, shift_frac, (scale_lo, scale_hi))
                # smooth_oracle is a metric test, not a model; it is the only
                # backend allowed to see the answer, and only via this attribute.
                if backend == "smooth_oracle":
                    seg_fn.gt = gt  # type: ignore[attr-defined]
                pred = np.asarray(seg_fn(img, pbox)) > 0
                if pred.shape != gt.shape:
                    raise ValueError(f"backend returned {pred.shape}, expected {gt.shape}")

                pred_path = ""
                if save_masks:
                    pred_path = (mask_dir / str(row["case_id"]).replace("/", "__")
                                 / f"{Path(row['image_path']).stem}_pred.png")
                    pred_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(pred_path), pred.astype(np.uint8) * 255)
                    pred_path = str(pred_path)

                rg, rp = roughness(gt), roughness(pred)
                rec = {
                    "case_id": row.get("case_id", ""),
                    "feature_name": row.get("feature_name", ""),
                    "modality": row.get("modality", ""),
                    "plane": row.get("plane", ""),
                    "slice_index": row.get("slice_index", ""),
                    "image_path": row["image_path"],
                    "mask_path": row["mask_path"],
                    "pred_mask_path": pred_path,
                    "backend": backend,
                    "model_id": model_id,
                    "gt_box": ",".join(map(str, gt_box)),
                    "prompt_box": ",".join(map(str, pbox)),
                    "prompt_box_iou": f"{box_iou(gt_box, pbox):.4f}",
                    "jitter_shift": shift_frac,
                    "jitter_scale": f"{scale_lo}-{scale_hi}",
                    "dice": f"{dice(gt, pred):.4f}",
                    "iou": f"{iou(gt, pred):.4f}",
                    "hd95": f"{hd95(gt, pred):.3f}",
                }
                rec.update({f"gt_{k}": f"{rg[k]:.4f}" for k in ROUGHNESS_KEYS})
                rec.update({f"pred_{k}": f"{rp[k]:.4f}" for k in ROUGHNESS_KEYS})
                writer.writerow(rec)
                n_ok += 1
                log.info("%s %s dice=%s", row.get("case_id", "?"), row.get("modality", ""),
                         rec["dice"])
            except Exception as e:  # noqa: BLE001 -- one bad row must not kill the sweep
                n_skip += 1
                log.warning("skip %s: %s", row.get("image_path", "?"), e)
            fh.flush()

    log.info("wrote %d row(s) (%d skipped) -> %s", n_ok, n_skip, out_path)


def _fmt(v: float, nd: int = 3, sign: bool = False) -> str:
    if v != v:  # NaN
        return "n/a"
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def evaluate(results: List[Path]) -> None:
    df = pd.concat([pd.read_csv(p) for p in results], ignore_index=True)
    num = ["dice", "iou", "hd95", "prompt_box_iou"] + \
          [f"gt_{k}" for k in ROUGHNESS_KEYS] + [f"pred_{k}" for k in ROUGHNESS_KEYS]
    for c in num:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for backend, g in df.groupby("backend"):
        print(f"\n{'=' * 68}\n{backend}   n={len(g)}   "
              f"model_id={g['model_id'].dropna().unique().tolist() or ['-']}\n{'=' * 68}")
        print(f"  prompt box IoU vs tight GT box (how hard the jitter was): "
              f"{_fmt(g['prompt_box_iou'].mean())}")
        print(f"  dice  mean {_fmt(g['dice'].mean())}   median {_fmt(g['dice'].median())}   "
              f"p10 {_fmt(g['dice'].quantile(0.10))}")
        print(f"  iou   mean {_fmt(g['iou'].mean())}")
        print(f"  hd95  mean {_fmt(g['hd95'].mean(), 2)} px   "
              f"median {_fmt(g['hd95'].median(), 2)} px")

        # The point of this directory. Dice can be high while these diverge.
        print("\n  Boundary character -- radiologist vs automatic:")
        print(f"    {'metric':<13}{'radiologist':>13}{'auto':>10}{'delta':>10}")
        for k in ("circularity", "solidity", "lobe_amp", "jag_amp", "elongation"):
            a, b = g[f"gt_{k}"].mean(), g[f"pred_{k}"].mean()
            print(f"    {k:<13}{_fmt(a):>13}{_fmt(b):>10}{_fmt(b - a, sign=True):>10}")
        frag = (g["pred_n_components"] > 1).mean()
        print(f"    fragmented predictions (>1 component): {frag:.1%}")

        # Interpretation, stated rather than left to the reader.
        d_circ = g["pred_circularity"].mean() - g["gt_circularity"].mean()
        d_jag = g["pred_jag_amp"].mean() - g["gt_jag_amp"].mean()
        if d_circ > 0.05 or d_jag < -0.02:
            print("\n  -> SMOOTHING DETECTED: the automatic contour is rounder / less "
                  "spiculated\n     than the radiologist's. Margin features (shape, margin "
                  "definition,\n     cortical breach) read off it will be biased toward the "
                  "smooth classes,\n     regardless of how good the Dice looks.")
        else:
            print("\n  -> no systematic smoothing at this sample size; boundary character "
                  "is preserved.")

        print(f"\n  Where real lesions sit on shape_probe's difficulty axis:\n"
              f"    radiologist lobe_amp {_fmt(g['gt_lobe_amp'].mean())}  "
              f"jag_amp {_fmt(g['gt_jag_amp'].mean())}\n"
              f"    (compare against shape_probe --difficulty: BASE lobe amplitude is 0.30 "
              f"at d=1.0)")

    if df["backend"].nunique() > 1:
        print(f"\n{'=' * 68}\nbackend comparison\n{'=' * 68}")
        piv = df.groupby("backend").agg(
            n=("dice", "size"), dice=("dice", "mean"), hd95=("hd95", "mean"),
            circ_delta=("pred_circularity", "mean"), jag_delta=("pred_jag_amp", "mean"))
        gt_circ = df.groupby("backend")["gt_circularity"].mean()
        gt_jag = df.groupby("backend")["gt_jag_amp"].mean()
        piv["circ_delta"] -= gt_circ
        piv["jag_delta"] -= gt_jag
        print(piv.round(3).to_string())
        print("\nRead this as a pair: high dice with a positive circ_delta / negative "
              "jag_delta\nmeans the region is right and the margin is wrong -- the failure "
              "mode that\nmatters here. box_fill is the floor; smooth_oracle is the metric "
              "check.")


def preview(results: List[Path], out: Path, n: int = 16, cols: int = 4,
            cell: int = 200, seed: int = 0) -> None:
    """Contact sheet: green = radiologist, red = automatic, yellow = prompt box.
    Numbers decide nothing until you have looked at the disagreements."""
    df = pd.concat([pd.read_csv(p) for p in results], ignore_index=True)
    df = df[df["pred_mask_path"].astype(str).str.strip() != ""]
    if df.empty:
        raise SystemExit("no rows with pred_mask_path -- re-run segment without --no-save-masks")
    df["dice"] = pd.to_numeric(df["dice"], errors="coerce")
    # Worst cases first: the mean tells you nothing you can act on.
    sample = df.nsmallest(min(n, len(df)), "dice")

    rows = int(np.ceil(len(sample) / cols))
    sheet = np.zeros((rows * cell, cols * cell, 3), dtype=np.uint8)
    for i, (_, r) in enumerate(sample.iterrows()):
        img = cv2.imread(str(r["image_path"]), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(str(r["mask_path"]), cv2.IMREAD_GRAYSCALE)
        pr = cv2.imread(str(r["pred_mask_path"]), cv2.IMREAD_GRAYSCALE)
        if img is None or gt is None or pr is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for m, col in ((gt, (0, 255, 0)), (pr, (0, 0, 255))):
            cnts, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(rgb, cnts, -1, col, 1)
        x0, y0, x1, y1 = (int(v) for v in str(r["prompt_box"]).split(","))
        cv2.rectangle(rgb, (x0, y0), (x1, y1), (0, 255, 255), 1)
        rgb = cv2.resize(rgb, (cell, cell), interpolation=cv2.INTER_NEAREST)
        cv2.putText(rgb, f"dice={r['dice']:.2f}", (4, cell - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255, 255, 255), 1, cv2.LINE_AA)
        y, x = (i // cols) * cell, (i % cols) * cell
        sheet[y:y + cell, x:x + cell] = rgb

    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    print(f"wrote {out}  ({len(sample)} worst-dice tiles; green=radiologist, "
          f"red=auto, yellow=prompt)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["segment", "eval", "preview"])
    ap.add_argument("--metadata", type=Path, help="preprocess metadata.csv built with --save-mask")
    ap.add_argument("--out", type=Path, help="results CSV (segment) or PNG (preview)")
    ap.add_argument("--results", type=Path, nargs="+", help="results CSV(s) for eval/preview")
    ap.add_argument("--backend", default="box_fill", choices=backends.available(),
                    help="box_fill = Dice floor; smooth_oracle = roughness-metric check; "
                         "the rest need weights")
    ap.add_argument("--model-id", default="", help="override the backend's default checkpoint")
    ap.add_argument("--shift-frac", type=float, default=0.10,
                    help="prompt-box centre jitter, as a fraction of box size. Sweep this to "
                         "find how accurate an upstream detector would need to be.")
    ap.add_argument("--scale-lo", type=float, default=0.90)
    ap.add_argument("--scale-hi", type=float, default=1.20)
    ap.add_argument("--no-jitter", action="store_true",
                    help="prompt with the exact GT box. Inflates Dice -- for diagnosis only, "
                         "never for a reported number.")
    ap.add_argument("--smooth-px", type=int, default=5, help="smooth_oracle kernel size")
    ap.add_argument("--no-save-masks", action="store_true",
                    help="skip writing predicted masks (disables --mode preview)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=16, help="tiles in the preview sheet")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.mode == "segment":
        if not args.metadata or not args.out:
            raise SystemExit("--metadata and --out are required for --mode segment")
        shift = 0.0 if args.no_jitter else args.shift_frac
        lo, hi = (1.0, 1.0) if args.no_jitter else (args.scale_lo, args.scale_hi)
        segment(args.metadata, args.out, args.backend, model_id=args.model_id,
                shift_frac=shift, scale_lo=lo, scale_hi=hi,
                save_masks=not args.no_save_masks, limit=args.limit, seed=args.seed,
                smooth_px=args.smooth_px)
    elif args.mode == "eval":
        if not args.results:
            raise SystemExit("--results is required for --mode eval")
        evaluate(args.results)
    else:
        if not args.results or not args.out:
            raise SystemExit("--results and --out are required for --mode preview")
        preview(args.results, args.out, n=args.n, seed=args.seed)


if __name__ == "__main__":
    main()
