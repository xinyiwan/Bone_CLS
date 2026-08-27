"""
Run the shape-perception probe: can MedGemma see an overlay drawn on the crop?

The model gets exactly the framing it gets in the real run -- an MRI crop of a
bone lesion with a red outline drawn on it -- but the outline is synthetic
instead of the true tumour contour, and the only question asked is what shape
it is. Two vocabularies, set at build time and read back off the metadata:

    icons     circle/square/triangle/star -- perception, chance 25%
    clinical  the 5 margin classes of feature_prompts.yaml -- discrimination,
              chance 20%, with --difficulty sweeping deformation amplitude so
              eval reports an accuracy CURVE rather than one number

Nothing else differs between them: same messages structure, same parser, same
sharding. See README.md for what each result means.

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
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "medgemma_pilot"))
import prompts  # noqa: E402  (only for messages_to_text -- prompt wording stays local)
import run_medgemma as mg  # noqa: E402  (path shim above must run first)

from shapes import SHAPE_SETS  # noqa: E402

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
    "background", "shape_set", "difficulty", "shape_params",
    "radius_px", "rotation_deg", "input_text", "raw_output",
    "thinking", "parsed_label", "reason", "shape", "correct", "model_id",
    "num_few_shot",
]

# --------------------------------------------------------------------------
# prompts -- one per shape set
# --------------------------------------------------------------------------
# The `icons` prompt asks a pure naming question. The `clinical` prompt
# deliberately mirrors the label_definitions in medgemma_pilot/feature_prompts.yaml
# (same discriminating axes: number of convex bulges, inward vs outward
# curvature, smooth vs jagged, one protrusion vs many), so a gap between this
# probe and the real run is attributable to the IMAGES, not the wording. If you
# retune the YAML definitions, retune these to match or the comparison breaks.

ICON_DEFS = "It is exactly one of: circle, square, triangle, star."

CLINICAL_DEFS = (
    "It is exactly one of these five margin descriptors:\n"
    "- round_oval: one smooth, continuous convex curve; no separate bulges.\n"
    "- lobulated: several (roughly 4-7) rounded convex lobes side by side, each "
    "smooth on its own, separated by shallow notches -- a cauliflower outline.\n"
    "- geographic: one broad, sharply demarcated CONCAVE arc, like a single bite "
    "scooped out of an otherwise smooth boundary.\n"
    "- irregular: many small jagged, angular projections scattered unpredictably; "
    "no countable or repeatable geometry.\n"
    "- exophytic: one single dominant protrusion sticking OUTWARD past an "
    "otherwise smooth boundary (mushroom-like / polypoid)."
)

SYSTEM_TEMPLATE = (
    "You are an expert radiologist reviewing an MRI image of a bone lesion.\n"
    "A single closed outline has been drawn on the image in RED.\n"
    "Your only task is to identify the shape of that red outline.\n"
    "{definitions}\n"
    "Judge the shape of the RED drawn outline itself -- not the anatomy, not the "
    "lesion, not any other structure in the image.\n"
    'Answer ONLY with a JSON object: {{"prediction": "<{options}>", '
    '"reason": "<one short sentence>"}}'
)

USER_TEMPLATE = (
    "This is a {modality} MRI of a bone lesion, {plane} plane. "
    "A red outline has been drawn on it. "
    "Which of these best describes the red outline: {options}?"
)

PROMPT_DEFS = {"icons": ICON_DEFS, "clinical": CLINICAL_DEFS}


def prompt_texts(shape_set: str) -> tuple:
    labels = SHAPE_SETS[shape_set]
    return (
        SYSTEM_TEMPLATE.format(definitions=PROMPT_DEFS[shape_set], options="|".join(labels)),
        USER_TEMPLATE.replace("{options}", ", ".join(labels)),
    )


def resolve_shape_set(name: str, df) -> str:
    """'auto' reads it off the metadata: the shape_set column if build_shapes
    wrote one, else whichever vocabulary the ground-truth labels belong to. This
    keeps a results CSV from being scored against the wrong chance level."""
    if name != "auto":
        return name
    if "shape_set" in df.columns:
        vals = {v for v in df["shape_set"].astype(str) if v in SHAPE_SETS}
        if len(vals) == 1:
            return vals.pop()
        if len(vals) > 1:
            raise SystemExit(f"metadata mixes shape sets {sorted(vals)}; build them separately")
    labels = set(df["shape"].astype(str))
    for key, names in SHAPE_SETS.items():
        if labels <= set(names):
            return key
    raise SystemExit(f"cannot infer --shape-set from labels {sorted(labels)}; pass it explicitly")


def build_messages(image, modality: str, plane: str, shape_set: str = "icons",
                   few_shot: Optional[List[tuple]] = None) -> List[dict]:
    """Same message structure as prompts.build_medgemma_messages (system turn
    with the constant task, then optional prior example turns, then one user turn
    with the query image) -- kept local and tiny because the probe's prompt is
    deliberately not feature-config driven.

    `few_shot` is [(PIL image, label), ...]. Each becomes a completed user ->
    assistant exchange before the query, with the assistant's reply in exactly
    the JSON format we ask for, so the examples demonstrate the output shape as
    well as the visual class."""
    system_text, user_text = prompt_texts(shape_set)
    query_text = user_text.format(modality=modality or "unknown-sequence",
                                  plane=plane or "unknown")

    messages: List[dict] = [{"role": "system", "content": [{"type": "text", "text": system_text}]}]
    for ex_image, ex_label in few_shot or []:
        messages.append({"role": "user", "content": [
            {"type": "image", "image": ex_image},
            {"type": "text", "text": query_text},
        ]})
        messages.append({"role": "assistant", "content": [
            {"type": "text",
             "text": f'{{"prediction": "{ex_label}", "reason": "reference example."}}'},
        ]})
    messages.append({"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": query_text},
    ]})
    return messages


def select_examples(df, labels: List[str], n_per_class: int, rng) -> List[dict]:
    """Pick n_per_class exemplar rows for every label, interleaved by class so the
    example turns cycle through the vocabulary rather than showing all of one
    class first.

    Exemplars are taken from the EASIEST difficulty available (largest value),
    because the point of an example is to show the prototype: on a sweep you want
    to ask "does seeing a pronounced lobulated margin help at d=0.35", not to
    spend the example on an ambiguous one. The caller must exclude the returned
    rows from inference -- scoring a model on an image it was just shown the
    answer to is not a measurement."""
    if n_per_class <= 0:
        return []
    pool = df
    if "difficulty" in df.columns:
        vals = pd.to_numeric(df["difficulty"], errors="coerce")
        if vals.notna().any():
            pool = df[vals == vals.max()]

    per_class: Dict[str, List[dict]] = {}
    for label in labels:
        rows = [r for _, r in pool.iterrows() if str(r.get("shape", "")) == label]
        if len(rows) < n_per_class:
            raise SystemExit(
                f"--num-few-shot {n_per_class} needs {n_per_class} example(s) of {label!r}, "
                f"but the metadata has only {len(rows)} at the easiest difficulty. "
                "Build more images, or lower --num-few-shot."
            )
        rng.shuffle(rows)
        per_class[label] = rows[:n_per_class]

    # Interleave: one of each class, then the next of each class.
    out: List[dict] = []
    for i in range(n_per_class):
        for label in labels:
            out.append(per_class[label][i])
    return out


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
    shape_set: str = "auto",
    batch_size: int = 1,
    shard_index: int = 0,
    num_shards: int = 1,
    limit: Optional[int] = None,
    num_few_shot: int = 0,
    few_shot_metadata: Optional[Path] = None,
    seed: int = 0,
) -> None:
    """One row per shape image. Rows are independent, so sharding is a plain
    strided split (every num_shards-th row) -- unlike run_medgemma there is no
    cross-row prompt context to preserve, and no post-hoc vote to protect.
    Each shard MUST get its own --out (main() adds the .shard<i> suffix)."""
    df = pd.read_csv(metadata, dtype=str).fillna("")
    if limit:
        df = df.head(limit)
    shape_set = resolve_shape_set(shape_set, df)
    labels = list(SHAPE_SETS[shape_set])
    log.info("shape set %r -> %d label(s), chance %.3f", shape_set, len(labels), 1 / len(labels))

    # Few-shot. Exemplars are chosen BEFORE sharding, from the unsharded frame,
    # so every shard shows the model the same examples -- otherwise the shards
    # would be running subtly different experiments and their CSVs could not be
    # concatenated. Loaded once here and reused for every query image.
    few_shot: List[tuple] = []
    if num_few_shot > 0:
        ex_df = pd.read_csv(few_shot_metadata, dtype=str).fillna("") if few_shot_metadata else df
        ex_rows = select_examples(ex_df, labels, num_few_shot, random.Random(seed))
        few_shot = [(mg.to_jpeg_rgb(r["image_path"]), str(r["shape"])) for r in ex_rows]
        if few_shot_metadata is None:
            # Same build: an exemplar image must not also be scored.
            ex_paths = {str(r["image_path"]) for r in ex_rows}
            before = len(df)
            df = df[~df["image_path"].astype(str).isin(ex_paths)]
            log.info("%d-shot per class (%d example turns); held %d exemplar image(s) out of inference",
                     num_few_shot, len(few_shot), before - len(df))
        else:
            log.info("%d-shot per class (%d example turns) from %s",
                     num_few_shot, len(few_shot), few_shot_metadata)

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
                messages = build_messages(image, row.get("modality", ""), row.get("plane", ""),
                                          shape_set=shape_set, few_shot=few_shot)
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
                label, reason = mg.parse_answer(raw, labels)
                gt = str(row.get("shape", ""))
                writer.writerow({
                    "case_id": row.get("case_id", ""),
                    "feature_name": row.get("feature_name", ""),
                    "modality": row.get("modality", ""),
                    "plane": row.get("plane", ""),
                    "image_path": row["image_path"],
                    "background": row.get("background", ""),
                    "shape_set": shape_set,
                    "difficulty": row.get("difficulty", ""),
                    "shape_params": row.get("shape_params", ""),
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
                    "num_few_shot": num_few_shot,
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

    shape_set = resolve_shape_set("auto", scored)
    labels = SHAPE_SETS[shape_set]

    acc = scored["correct"].astype(float).mean()
    print(f"\nShape probe [{shape_set}]: {n} images, accuracy {acc:.3f}  "
          f"(chance = {1/len(labels):.3f})")
    n_fail = len(df) - n
    if n_fail:
        print(f"  unparseable answers (excluded): {n_fail} ({n_fail/len(df):.1%})")

    print("\nPer-shape recall:")
    for shape, g in scored.groupby("shape"):
        print(f"  {shape:11s} n={len(g):4d}  acc={g['correct'].astype(float).mean():.3f}")

    # The point of the clinical set: accuracy as a FUNCTION of deformation
    # amplitude. A curve that falls to chance between two levels localises the
    # model's discrimination threshold, which can then be compared against the
    # amplitude actually present in the annotated lesions.
    if "difficulty" in scored and scored["difficulty"].astype(str).str.strip().any():
        d = scored[scored["difficulty"].astype(str).str.strip() != ""].copy()
        d["difficulty"] = d["difficulty"].astype(float)
        if d["difficulty"].nunique() > 1:
            print("\nAccuracy by difficulty (deformation amplitude; lower = subtler):")
            pivot = (d.assign(correct=d["correct"].astype(float))
                       .pivot_table(index="shape", columns="difficulty",
                                    values="correct", aggfunc="mean"))
            print(pivot.round(3).to_string())
            print("\n  overall:")
            for lv, g in d.groupby("difficulty"):
                print(f"    d={lv:<5g} n={len(g):4d}  acc={g['correct'].astype(float).mean():.3f}")

    print("\nPrediction distribution (a flat-guessing model collapses onto one label):")
    for label, cnt in scored["parsed_label"].value_counts().items():
        print(f"  {label:14s} {cnt:4d}  ({cnt/n:.1%})")

    print("\nConfusion matrix (rows = true, cols = predicted):")
    print(pd.crosstab(scored["shape"], scored["parsed_label"]).to_string())

    for col in ("num_few_shot", "background", "modality", "plane", "model_id"):
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
    ap.add_argument("--shape-set", default="auto", choices=["auto", *sorted(SHAPE_SETS)],
                    help="which label vocabulary and prompt to use; 'auto' reads it off the "
                         "metadata's shape_set column (or infers it from the labels)")
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
    # few-shot
    ap.add_argument("--num-few-shot", type=int, default=0,
                    help="N labeled example turns PER CLASS before the query image "
                         "(0 = zero-shot). 1 with the clinical set = 5 examples, which is where "
                         "few-shot pays off: the 5 margin classes are far easier to pin down by "
                         "example than by prose. Exemplars are taken from the easiest difficulty "
                         "and held out of scoring.")
    ap.add_argument("--few-shot-metadata", type=Path, default=None,
                    help="take exemplars from a DIFFERENT build (e.g. the blank-background or "
                         "easy-difficulty one) instead of the images being scored. Nothing is then "
                         "held out of --metadata, so zero-shot and few-shot runs score the exact "
                         "same image set and are directly comparable.")
    ap.add_argument("--seed", type=int, default=0, help="exemplar selection seed")
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
        infer(args.metadata, out, generate, model_id=args.model_id, shape_set=args.shape_set,
              batch_size=args.batch_size,
              shard_index=args.shard_index, num_shards=args.num_shards, limit=args.limit,
              num_few_shot=args.num_few_shot, few_shot_metadata=args.few_shot_metadata,
              seed=args.seed)
    else:
        if not args.results:
            raise SystemExit("--results is required for --mode eval")
        evaluate(args.results)


if __name__ == "__main__":
    main()
