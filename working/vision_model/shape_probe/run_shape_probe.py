"""
Run the shape-perception probe: can MedGemma see an overlay drawn on the crop?

The model gets exactly the framing it gets in the real run -- an MRI crop of a
bone lesion with a red outline drawn on it -- but the outline is a circle,
square, triangle or star instead of the true tumour contour, and the only
question asked is which shape it is. Chance is 25%.

Reads the CSV from `build_shapes.py`, writes one row per image, and scores it.
It reuses `medgemma_pilot/run_medgemma.py` for model loading, batched
generation, backend selection, thinking-block stripping and JSON answer parsing,
so the decoding path is IDENTICAL to the real experiment. Nothing in
medgemma_pilot/ or preprocess/ is modified.

Throughput flags mirror run_medgemma.py exactly (--backend / --batch-size /
--num-shards / --shard-index), so the same one-process-per-GPU SLURM harness
drives both. There is no `aggregate` step: the probe scores per image, not per
lesion, so shard CSVs are just concatenated -- `--mode eval` takes several.

    # 1. infer (resume-safe: re-run to continue an interrupted job)
    python run_shape_probe.py --mode infer --model-id /models/medgemma-1.5-4b-it \
        --metadata /results/shape_probe/mri/shape_metadata.csv \
        --batch-size 16 --out /results/shape_probe/mri/probe_results.csv

    # 2. score (pass every shard file from a multi-GPU run)
    python run_shape_probe.py --mode eval \
        --results /results/shape_probe/mri/probe_results.shard*.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "medgemma_pilot"))
import prompts  # noqa: E402  (only for messages_to_text -- prompt wording stays local)
import run_medgemma as mg  # noqa: E402  (path shim above must run first)

from shapes import SHAPES  # noqa: E402

log = logging.getLogger("shape_probe")

# `image_path` is the shape image -- unlike the real run there is no plain/overlay
# swap, so it is both the resume key and what the model saw. `shape` is the
# ground truth. `input_text` is the fully rendered prompt (same column and same
# renderer as run_medgemma.py, so both CSVs are readable the same way) -- without
# it a surprising result can't be traced back to what was actually asked. Run
# config (model_id/background) is stamped on every row so a results CSV is
# self-describing and shard files concatenate cleanly.
RESULT_FIELDS = [
    "case_id", "feature_name", "modality", "plane", "image_path",
    "background", "radius_px", "rotation_deg", "input_text", "raw_output",
    "thinking", "parsed_label", "reason", "shape", "correct", "model_id",
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


def _done_keys(out_path: Path) -> set:
    """Already-inferred image paths, so re-running resumes instead of duplicating."""
    if not out_path.exists():
        return set()
    df = pd.read_csv(out_path, dtype=str).fillna("")
    if df.empty or "image_path" not in df.columns:
        return set()
    return set(df["image_path"])


def _check_resumable(out_path: Path) -> None:
    """Fail loudly when --out exists but has a different column set: we append
    with a fixed DictWriter fieldname list, so a mismatched header would silently
    write values under the wrong headings. (Same guard as run_medgemma.)"""
    if not out_path.exists() or out_path.stat().st_size == 0:
        return
    with open(out_path, newline="") as fh:
        header = next(csv.reader(fh), [])
    if header != RESULT_FIELDS:
        raise SystemExit(
            f"{out_path} was written with a different schema, so this run cannot append to it.\n"
            f"  missing column(s): {[c for c in RESULT_FIELDS if c not in header] or 'none'}\n"
            f"  extra column(s):   {[c for c in header if c not in RESULT_FIELDS] or 'none'}\n"
            "Pass a fresh --out path."
        )


def infer(
    metadata: Path,
    out_path: Path,
    generate: "mg.Generate",
    model_id: str = "",
    batch_size: int = 1,
    shard_index: int = 0,
    num_shards: int = 1,
    limit: Optional[int] = None,
) -> None:
    """One row per shape image. Rows are independent, so sharding is a plain
    strided split (every num_shards-th row) -- unlike run_medgemma there is no
    cross-row prompt context to preserve, and no post-hoc vote to protect.
    Each shard MUST get its own --out (main() adds the .shard<i> suffix)."""
    df = pd.read_csv(metadata, dtype=str).fillna("")
    if limit:
        df = df.head(limit)
    n_total = len(df)
    if num_shards > 1:
        df = df.iloc[shard_index::num_shards]
        log.info("shard %d/%d -> %d of %d metadata row(s)", shard_index, num_shards, len(df), n_total)

    _check_resumable(out_path)
    done = _done_keys(out_path)
    tasks = [r for _, r in df.iterrows() if str(r["image_path"]) not in done]
    n_batches = (len(tasks) + batch_size - 1) // max(batch_size, 1)
    log.info("%d image(s) in shard; %d already done -> %d to infer in %d batch(es) of up to %d",
             len(df), len(df) - len(tasks), len(tasks), n_batches, batch_size)
    if not tasks:
        return

    new_file = not out_path.exists()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_FIELDS)
        if new_file:
            writer.writeheader()
            fh.flush()

        t_start = time.perf_counter()
        n_done = 0
        for bi in range(n_batches):
            chunk = tasks[bi * batch_size:(bi + 1) * batch_size]
            batch_messages: List[List[dict]] = []
            batch_rows = []
            for row in chunk:
                try:
                    image = mg.to_jpeg_rgb(row["image_path"])
                except Exception as e:  # noqa: BLE001 -- missing/unreadable image
                    log.warning("skip image %s: %s", row["image_path"], e)
                    continue
                messages = build_messages(image, row.get("modality", ""), row.get("plane", ""))
                batch_messages.append(messages)
                # Render now, while the messages exist -- the image becomes an
                # '<image>' placeholder, so this is cheap to store per row.
                batch_rows.append((row, prompts.messages_to_text(messages)))
            if not batch_messages:
                continue

            raws = generate(batch_messages)
            if len(raws) != len(batch_rows):  # a backend that drops/reorders would misattribute every row
                raise RuntimeError(
                    f"backend returned {len(raws)} output(s) for a batch of {len(batch_rows)} prompt(s); "
                    "outputs must be one-per-prompt and in order"
                )

            for (row, input_text), raw in zip(batch_rows, raws):
                label, reason = mg.parse_answer(raw, list(SHAPES))
                gt = str(row.get("shape", ""))
                writer.writerow({
                    "case_id": row.get("case_id", ""),
                    "feature_name": row.get("feature_name", ""),
                    "modality": row.get("modality", ""),
                    "plane": row.get("plane", ""),
                    "image_path": row["image_path"],
                    "background": row.get("background", ""),
                    "radius_px": row.get("radius_px", ""),
                    "rotation_deg": row.get("rotation_deg", ""),
                    "input_text": input_text,
                    "raw_output": raw,
                    "thinking": mg.extract_thinking(raw),
                    "parsed_label": label,
                    "reason": reason,
                    "shape": gt,
                    "correct": int(label == gt) if label != "PARSE_FAILED" else "",
                    "model_id": model_id,
                })
                log.info("%s [%s] -> %s (true=%s)", row.get("case_id", "?"),
                         row.get("background", "?"), label, gt)
            fh.flush()  # resume-safe at batch granularity

            n_done += len(batch_rows)
            rate = n_done / max(time.perf_counter() - t_start, 1e-9)
            log.info("batch %d/%d done -- %d/%d image(s), %.2f img/s, ~%.1f min left",
                     bi + 1, n_batches, n_done, len(tasks), rate,
                     (len(tasks) - n_done) / rate / 60 if rate > 0 else float("nan"))

    log.info("done -> %s", out_path)


def evaluate(results: List[Path] | Path) -> None:
    """Score one or more results CSVs (pass every shard file from a multi-GPU
    run; the probe has no per-lesion vote, so concatenating is the aggregation)."""
    paths = [results] if isinstance(results, Path) else list(results)
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    if len(paths) > 1:
        print(f"combined {len(paths)} results file(s) -> {len(df)} rows")

    scored = df[df["parsed_label"] != "PARSE_FAILED"]
    n = len(scored)
    if n == 0:
        print("no scorable rows")
        return

    acc = scored["correct"].astype(float).mean()
    print(f"\nShape probe: {n} images, accuracy {acc:.3f}  (chance = {1/len(SHAPES):.3f})")
    n_fail = len(df) - n
    if n_fail:
        print(f"  unparseable answers (excluded): {n_fail} ({n_fail/len(df):.1%})")

    print("\nPer-shape recall:")
    for shape, g in scored.groupby("shape"):
        print(f"  {shape:9s} n={len(g):4d}  acc={g['correct'].astype(float).mean():.3f}")

    print("\nPrediction distribution (a flat-guessing model collapses onto one label):")
    for label, cnt in scored["parsed_label"].value_counts().items():
        print(f"  {label:14s} {cnt:4d}  ({cnt/n:.1%})")

    print("\nConfusion matrix (rows = true, cols = predicted):")
    print(pd.crosstab(scored["shape"], scored["parsed_label"]).to_string())

    for col in ("background", "modality", "plane", "model_id"):
        if col in scored and scored[col].nunique() > 1:
            print(f"\nAccuracy by {col}:")
            for key, g in scored.groupby(col):
                print(f"  {str(key):28s} n={len(g):4d}  acc={g['correct'].astype(float).mean():.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["infer", "eval"])
    ap.add_argument("--metadata", type=Path, help="shape_metadata.csv from build_shapes.py (infer mode)")
    ap.add_argument("--out", type=Path, help="output CSV (infer mode)")
    ap.add_argument("--results", type=Path, nargs="+",
                    help="results CSV(s) (eval mode); pass every shard file from a multi-GPU run")
    # model / decoding -- kept identical to run_medgemma.py so both share a launcher
    ap.add_argument("--model-id", default="google/medgemma-1.5-4b-it")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="images per generate() call; 8-32 for the 4B on an 80GB A100 (hf), 256+ for vllm")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1,
                    help="split the metadata across N processes (one per GPU); each shard gets its "
                         "own --out (.shard<i> suffix), pass them all to --mode eval afterwards")
    ap.add_argument("--limit", type=int, default=None, help="only the first N metadata rows (smoke test)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit(f"--shard-index must be in [0, {args.num_shards}) for --num-shards {args.num_shards}")

    if args.mode == "infer":
        if not args.metadata or not args.out:
            raise SystemExit("--metadata and --out are required for --mode infer")
        out = args.out
        if args.num_shards > 1:
            # Shards append concurrently; sharing one --out would interleave
            # half-written rows. Give each its own file, concatenate at eval.
            out = out.with_name(f"{out.stem}.shard{args.shard_index}{out.suffix}")
            log.info("sharded run -> writing %s", out)
        generate = mg.make_generate(args.backend, args.model_id, args.max_new_tokens)
        infer(args.metadata, out, generate, model_id=args.model_id, batch_size=args.batch_size,
              shard_index=args.shard_index, num_shards=args.num_shards, limit=args.limit)
    else:
        if not args.results:
            raise SystemExit("--results is required for --mode eval")
        evaluate(args.results)


if __name__ == "__main__":
    main()
