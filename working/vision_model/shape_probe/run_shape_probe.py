"""
Run the shape-perception probe: can MedGemma see an overlay drawn on the crop?

The model gets exactly the framing it gets in the real run -- an MRI crop of a
bone lesion with a red outline drawn on it -- but the outline is a circle,
square, triangle or star instead of the true tumour contour, and the only
question asked is which shape it is. Chance is 25%.

Reads the CSV from `build_shapes.py`, writes one row per image, and scores it.
It reuses `medgemma_pilot.run_medgemma` for model loading, generation, thinking-
block stripping and JSON answer parsing, so the decoding path is IDENTICAL to
the real experiment. Nothing in medgemma_pilot/ or preprocess/ is modified.

    # 1. infer (resume-safe: re-run to continue an interrupted job)
    python run_shape_probe.py --mode infer --model-id /models/medgemma-1.5-4b-it \
        --metadata /results/shape_probe/mri/shape_metadata.csv \
        --out /results/shape_probe/mri/probe_results.csv

    # 2. score
    python run_shape_probe.py --mode eval --results /results/shape_probe/mri/probe_results.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "medgemma_pilot"))
import run_medgemma as mg  # noqa: E402  (path shim above must run first)

from shapes import SHAPES  # noqa: E402

log = logging.getLogger("shape_probe")

RESULT_FIELDS = [
    "case_id", "feature_name", "modality", "plane", "image_path",
    "background", "radius_px", "raw_output", "thinking",
    "parsed_label", "shape", "correct", "model_id",
]

SYSTEM_TEXT = (
    "You are an expert radiologist reviewing an MRI image of a bone lesion.\n"
    "A single geometric outline has been drawn on the image in RED.\n"
    "Your only task is to identify which geometric shape that red outline is.\n"
    "It is exactly one of: circle, square, triangle, star.\n"
    "Judge the shape of the RED drawn outline itself -- not the anatomy, not the "
    "lesion, not any other structure in the image.\n"
    'Answer ONLY with a JSON object: {"prediction": "<circle|square|triangle|star>", '
    '"reason": "<one short sentence>"}'
)

USER_TEXT = (
    "This is a {modality} MRI of a bone lesion, {plane} plane. "
    "A red geometric outline has been drawn on it. "
    "Which shape is the red outline: circle, square, triangle, or star?"
)


def build_messages(image, modality: str, plane: str) -> List[dict]:
    """Same message structure as prompts.build_medgemma_messages (system turn
    with the constant task, one user turn with image + context) -- kept local and
    tiny because the probe's prompt is deliberately not feature-config driven."""
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_TEXT}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": USER_TEXT.format(modality=modality or "unknown-sequence",
                                                      plane=plane or "unknown")},
        ]},
    ]


def _done_paths(out_path: Path) -> set:
    """Already-inferred image paths, for resume."""
    if not out_path.exists():
        return set()
    try:
        done = pd.read_csv(out_path)
    except Exception:  # noqa: BLE001 -- truncated CSV from a hard kill
        return set()
    return set(done["image_path"].astype(str)) if "image_path" in done else set()


def infer(metadata: Path, out_path: Path, model_id: str, max_new_tokens: int = 512,
          limit: Optional[int] = None) -> None:
    df = pd.read_csv(metadata)
    if limit:
        df = df.head(limit)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_paths(out_path)
    if done:
        log.info("resuming: %d image(s) already done", len(done))
    todo = df[~df["image_path"].astype(str).isin(done)]
    log.info("%d image(s) to infer (of %d)", len(todo), len(df))
    if todo.empty:
        return

    generate = mg.make_hf_generate(model_id, max_new_tokens)

    write_header = not out_path.exists() or not done
    with open(out_path, "a" if done else "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_FIELDS)
        if write_header:
            writer.writeheader()
            fh.flush()

        for n, (_, row) in enumerate(todo.iterrows(), 1):
            img_path = str(row["image_path"])
            try:
                image = mg.to_jpeg_rgb(img_path)
                raw = generate(build_messages(image, str(row.get("modality", "")),
                                              str(row.get("plane", ""))))
                pred, _reason = mg.parse_answer(raw, list(SHAPES))
                thinking = mg.extract_thinking(raw)
            except Exception as e:  # noqa: BLE001 -- never lose a whole run to one image
                log.exception("failed on %s: %s", img_path, e)
                raw, pred, thinking = f"ERROR: {e}", "ERROR", ""

            gt = str(row.get("shape", ""))
            writer.writerow({
                "case_id": row.get("case_id", ""),
                "feature_name": row.get("feature_name", ""),
                "modality": row.get("modality", ""),
                "plane": row.get("plane", ""),
                "image_path": img_path,
                "background": row.get("background", ""),
                "radius_px": row.get("radius_px", ""),
                "raw_output": raw,
                "thinking": thinking,
                "parsed_label": pred,
                "shape": gt,
                "correct": int(pred == gt),
                "model_id": model_id,
            })
            fh.flush()
            if n % 10 == 0 or n == len(todo):
                log.info("  %d/%d", n, len(todo))

    log.info("wrote %s", out_path)


def evaluate(results: Path) -> None:
    df = pd.read_csv(results)
    scored = df[df["parsed_label"] != "ERROR"]
    n = len(scored)
    if n == 0:
        print("no scorable rows")
        return

    acc = scored["correct"].mean()
    print(f"\nShape probe: {n} images, accuracy {acc:.3f}  (chance = {1/len(SHAPES):.3f})")
    n_fail = int((scored["parsed_label"] == "PARSE_FAILED").sum())
    if n_fail:
        print(f"  unparseable answers: {n_fail} ({n_fail/n:.1%})")

    print("\nPer-shape recall:")
    for shape, g in scored.groupby("shape"):
        print(f"  {shape:9s} n={len(g):4d}  acc={g['correct'].mean():.3f}")

    print("\nPrediction distribution (a flat-guessing model collapses onto one label):")
    for label, cnt in scored["parsed_label"].value_counts().items():
        print(f"  {label:14s} {cnt:4d}  ({cnt/n:.1%})")

    print("\nConfusion matrix (rows = true, cols = predicted):")
    print(pd.crosstab(scored["shape"], scored["parsed_label"]).to_string())

    for col in ("background", "modality", "plane"):
        if col in scored and scored[col].nunique() > 1:
            print(f"\nAccuracy by {col}:")
            for key, g in scored.groupby(col):
                print(f"  {str(key):22s} n={len(g):4d}  acc={g['correct'].mean():.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["infer", "eval"])
    ap.add_argument("--metadata", type=Path, help="shape_metadata.csv from build_shapes.py")
    ap.add_argument("--out", type=Path, help="results CSV (infer)")
    ap.add_argument("--results", type=Path, help="results CSV (eval)")
    ap.add_argument("--model-id", default="google/medgemma-1.5-4b-it")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.mode == "infer":
        if not args.metadata or not args.out:
            ap.error("--mode infer needs --metadata and --out")
        infer(args.metadata, args.out, args.model_id, args.max_new_tokens, args.limit)
    else:
        if not args.results:
            ap.error("--mode eval needs --results")
        evaluate(args.results)


if __name__ == "__main__":
    main()
