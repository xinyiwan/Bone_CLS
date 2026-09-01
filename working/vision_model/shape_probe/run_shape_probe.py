"""
Run the shape-perception probe: can MedGemma see an overlay drawn on the crop?

The model gets exactly the framing it gets in the real run -- an MRI crop of a
bone lesion with a red outline drawn on it -- but the outline is synthetic
instead of the true tumour contour, and the only question asked is what shape
it is. Two vocabularies, set at build time and read back off the metadata:

    icons     circle/square/triangle/star -- perception, chance 25%
    clinical  the margin classes of feature_prompts.yaml -- discrimination, with
              --difficulty sweeping deformation amplitude so eval reports an
              accuracy CURVE rather than one number. Which classes, and hence the
              chance level, is read off the images: build_shapes.py --skip-shapes
              drops geographic/exophytic by default, giving 3 classes at 33%

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
    # 1 when generation hit --max-new-tokens mid-thought, so no answer was ever
    # emitted. Recorded per row because the truncation RATE is a property of the
    # token budget, not of the model's ability: a run with 30% truncation is
    # measuring the budget. Older CSVs predate this column; `--mode reparse`
    # backfills it from raw_output.
    "truncated",
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

# One definition line per label, keyed so a build that omits a class also omits
# its definition. Offering a label the images never contain is not a harmless
# extra option: it is a wrong answer the prompt itself invites, and it inflates
# the apparent number of alternatives the chance level is computed against.
CLINICAL_DEF_LINES = {
    "round_oval": "- round_oval: one smooth, continuous convex curve; no separate bulges.",
    "lobulated": "- lobulated: an overall smooth, oval-ish outline that gently waves in and out "
                 "-- a few (roughly 4-7) broad, shallow rounded lobes riding on the curve, each "
                 "smooth on its own, separated by soft shallow notches.",
    "geographic": "- geographic: one broad, sharply demarcated CONCAVE arc, like a single bite "
                  "scooped out of an otherwise smooth boundary.",
    "irregular": "- irregular: a few patches of sharp, jagged, angular projections at "
                 "unpredictable places on the boundary, with smoother stretches between them; "
                 "no countable or repeatable geometry.",
    "exophytic": "- exophytic: one single dominant protrusion sticking OUTWARD past an "
                 "otherwise smooth boundary (mushroom-like / polypoid).",
}


def definitions_text(shape_set: str, labels: List[str]) -> str:
    """The definitions block for exactly `labels`, in the set's canonical order."""
    if shape_set == "icons":
        return f"It is exactly one of: {', '.join(labels)}."
    missing = [l for l in labels if l not in CLINICAL_DEF_LINES]
    if missing:
        raise SystemExit(f"no definition line for clinical shape(s) {missing}; add one to "
                         "CLINICAL_DEF_LINES before probing them")
    head = f"It is exactly one of these {len(labels)} margin descriptors:"
    return "\n".join([head, *(CLINICAL_DEF_LINES[l] for l in labels)])


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

# One reason per label, used as the assistant reply in few-shot example turns.
# A constant string ("reference example.") teaches the model that the `reason`
# field is filler, which is worse than useless here: the reason is the only place
# the model states WHICH cue it used, so a filler exemplar both wastes the
# demonstration and makes the free-text column unusable for error analysis. Each
# string below names the single discriminating cue for its class in the same
# vocabulary as the definitions above, so the examples reinforce the rubric
# instead of fighting it. Keep them one short sentence, cue-only, and never
# mention difficulty or anatomy -- an exemplar reason is a template to copy.
REFERENCE_REASONS = {
    # icons
    "circle": "The outline is a single smooth closed curve with no corners.",
    "square": "The outline has four straight sides meeting at four corners.",
    "triangle": "The outline has three straight sides meeting at three corners.",
    "star": "The outline alternates sharp outward points with deep inward notches.",
    # clinical
    "round_oval": "The outline is one smooth continuous convex curve with no separate bulges.",
    "lobulated": "The outline is broadly oval but waves gently in and out over a few broad shallow lobes.",
    "geographic": "The outline is smooth except for one broad, sharply demarcated concave arc scooped inward.",
    "irregular": "The outline has jagged angular projections in a few unpredictable places with smoother stretches between.",
    "exophytic": "The outline is smooth except for one dominant protrusion sticking outward from the boundary.",
}


def reference_reason(label: str) -> str:
    """Exemplar `reason` text for `label`. Missing entries fail loudly rather
    than silently reintroducing a filler reason: a new shape class must get its
    own cue sentence or its few-shot turn teaches nothing."""
    try:
        return REFERENCE_REASONS[label]
    except KeyError:
        raise SystemExit(
            f"no REFERENCE_REASONS entry for shape {label!r}; add a one-sentence cue "
            "for it in run_shape_probe.py before using it as a few-shot exemplar"
        ) from None


def prompt_texts(shape_set: str, labels: Optional[List[str]] = None) -> tuple:
    """(system_text, user_template) offering exactly `labels` (default: the whole
    set). Pass the classes the build actually contains -- see active_labels."""
    labels = list(labels or SHAPE_SETS[shape_set])
    return (
        SYSTEM_TEMPLATE.format(definitions=definitions_text(shape_set, labels),
                               options="|".join(labels)),
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


def active_labels(shape_set: str, df) -> List[str]:
    """The classes the data actually contains, in the set's canonical order.

    `build_shapes.py --skip-shapes` can leave a class out of a build, so the
    vocabulary of a run is a property of its IMAGES, not of SHAPE_SETS. Reading it
    off the ground truth here means the prompt offers only answerable options and
    the chance level matches the real number of alternatives -- and it needs no
    flag, so a build and its probe cannot drift apart."""
    present = {str(v) for v in df["shape"]}
    labels = [n for n in SHAPE_SETS[shape_set] if n in present]
    if not labels:
        raise SystemExit(f"no {shape_set!r} labels in the metadata's `shape` column "
                         f"(found {sorted(present)})")
    dropped = [n for n in SHAPE_SETS[shape_set] if n not in present]
    if dropped:
        log.info("classes absent from this build and therefore NOT offered: %s", dropped)
    return labels


def build_messages(image, modality: str, plane: str, shape_set: str = "icons",
                   few_shot: Optional[List[tuple]] = None,
                   labels: Optional[List[str]] = None) -> List[dict]:
    """Same message structure as prompts.build_medgemma_messages (system turn
    with the constant task, then optional prior example turns, then one user turn
    with the query image) -- kept local and tiny because the probe's prompt is
    deliberately not feature-config driven.

    `few_shot` is [(PIL image, label), ...]. Each becomes a completed user ->
    assistant exchange before the query, with the assistant's reply in exactly
    the JSON format we ask for, so the examples demonstrate the output shape as
    well as the visual class. The example's reason is that class's own
    discriminating cue (REFERENCE_REASONS), not a constant: a filler reason
    teaches the model the field is decorative, which is exactly the wrong lesson
    when `reason` is the only trace of what cue it used."""
    system_text, user_text = prompt_texts(shape_set, labels)
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
             "text": f'{{"prediction": "{ex_label}", '
                     f'"reason": "{reference_reason(ex_label)}"}}'},
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


def show_exemplars(metadata: Path, shape_set: str, n_per_class: int, seed: int,
                   sheet: Optional[Path] = None) -> None:
    """Print (and optionally tile) the exact exemplars a --mode infer run with the
    same --few-shot-metadata/--num-few-shot/--seed would show the model.

    Few-shot exemplars are the one part of the prompt nobody sees in the results
    CSV as an image, so a mislabeled or atypical example silently degrades every
    row. This makes them inspectable without spending a GPU: same select_examples
    call, same seeded RNG, so the listing IS what inference will use."""
    df = pd.read_csv(metadata, dtype=str).fillna("")
    shape_set = resolve_shape_set(shape_set, df)
    labels = active_labels(shape_set, df)
    rows = select_examples(df, labels, n_per_class, random.Random(seed))

    print(f"\n{len(rows)} exemplar turn(s) [{shape_set}], seed {seed}, from {metadata}")
    for i, r in enumerate(rows, 1):
        print(f"\n  {i}. {r['shape']}  (difficulty={r.get('difficulty', '') or '-'}, "
              f"params={r.get('shape_params', '') or '-'})")
        print(f"     image:  {r['image_path']}")
        print(f"     reason: {reference_reason(str(r['shape']))}")

    if sheet:
        # Reuse preview.py's contact sheet on just these rows, so the captions and
        # sizing match the QC sheet for the build itself.
        import preview  # local, optional: only this branch needs cv2

        tmp = sheet.with_suffix(".exemplars.csv")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(tmp, index=False)
        preview.contact_sheet(tmp, sheet, n=len(rows), cols=min(len(labels), 6))


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
    # Two label lists, deliberately different:
    #   `labels`  -- what the prompt OFFERS: only the classes this build contains.
    #   `vocab`   -- what the parser ACCEPTS: the whole set. An answer naming an
    #               unoffered class is then recorded as a wrong prediction, not as
    #               PARSE_FAILED, so the failure stays visible in the confusion
    #               matrix instead of vanishing from the denominator.
    labels = active_labels(shape_set, df)
    vocab = list(SHAPE_SETS[shape_set])
    log.info("shape set %r -> %d label(s) %s, chance %.3f",
             shape_set, len(labels), labels, 1 / len(labels))

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
                                          shape_set=shape_set, few_shot=few_shot, labels=labels)
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
                label, reason = mg.parse_answer(raw, vocab)
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
                    "truncated": int(mg.was_truncated(raw)),
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


def reparse(results: List[Path] | Path, shape_set: str = "auto", write: bool = False) -> None:
    """Re-derive `parsed_label` / `correct` / `truncated` from the stored
    `raw_output`, without a GPU.

    Every run keeps the model's full raw text, so parsing is a pure function of
    data already on disk. When the parser is fixed, the fix can be applied
    retroactively to every result ever produced -- no re-inference, no new
    GPU-hours, and the before/after diff quantifies exactly how much the old
    parse distorted the numbers.

    Prints the change breakdown and only rewrites the CSVs with --write, so the
    default is a dry run you can read before committing to it.
    """
    paths = [results] if isinstance(results, Path) else list(results)
    total = changed = 0
    moved: Dict[str, int] = {}
    for path in paths:
        df = pd.read_csv(path).fillna("")
        if "raw_output" not in df.columns:
            raise SystemExit(f"{path} has no raw_output column; nothing to reparse")
        vocab = list(SHAPE_SETS[resolve_shape_set(shape_set, df)])
        new_label, new_reason, new_trunc = [], [], []
        for _, row in df.iterrows():
            raw = str(row["raw_output"])
            lab, rsn = mg.parse_answer(raw, vocab)
            new_label.append(lab)
            new_reason.append(rsn)
            new_trunc.append(int(mg.was_truncated(raw)))
        old = df["parsed_label"].astype(str).tolist()
        for o, n in zip(old, new_label):
            total += 1
            if o != n:
                changed += 1
                moved[f"{o} -> {n}"] = moved.get(f"{o} -> {n}", 0) + 1
        gt = df["shape"].astype(str)
        df["parsed_label"] = new_label
        df["reason"] = new_reason
        df["truncated"] = new_trunc
        df["correct"] = [int(l == g) if l != "PARSE_FAILED" else ""
                         for l, g in zip(new_label, gt)]
        if write:
            df.to_csv(path, index=False)

    print(f"\nreparsed {total} row(s) across {len(paths)} file(s); "
          f"{changed} label(s) changed ({changed/max(total,1):.1%})")
    for k, v in sorted(moved.items(), key=lambda kv: -kv[1]):
        print(f"  {v:>5}  {k}")
    print("\n(dry run -- pass --write to update the CSVs)" if not write
          else "\nCSVs updated in place.")


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
    # Chance is 1/(classes the run actually offered), not 1/(size of the
    # vocabulary): a --skip-shapes build of 3 classes has chance 0.333, and
    # scoring it against 0.200 would read as a real effect.
    labels = active_labels(shape_set, scored)

    acc = scored["correct"].astype(float).mean()
    print(f"\nShape probe [{shape_set}]: {n} images, accuracy {acc:.3f}  "
          f"(chance = {1/len(labels):.3f})")
    n_fail = len(df) - n
    if n_fail:
        print(f"  unparseable answers (excluded): {n_fail} ({n_fail/len(df):.1%})")
    # Truncation is a property of --max-new-tokens, not of the model's ability to
    # see shapes, so it belongs next to the accuracy rather than buried. A high
    # rate means the headline number is measuring the token budget: raise
    # --max-new-tokens and re-run before reading anything into the classes.
    if "truncated" in df.columns:
        n_trunc = int(pd.to_numeric(df["truncated"], errors="coerce").fillna(0).sum())
        if n_trunc:
            print(f"  generations cut off mid-thought: {n_trunc} ({n_trunc/len(df):.1%})"
                  f" -- raise --max-new-tokens (currently the answer is never emitted"
                  f" for these)")

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
    ap.add_argument("--mode", required=True,
                    choices=["infer", "eval", "exemplars", "reparse"],
                    help="'exemplars' is a no-GPU dry run: print (and optionally tile) the "
                         "few-shot examples the same flags would send to the model. "
                         "'reparse' re-derives parsed_label/correct/truncated from the "
                         "stored raw_output of an existing --results CSV, also without a "
                         "GPU -- use it to apply a parser fix retroactively")
    ap.add_argument("--metadata", type=Path, help="shape_metadata.csv from build_shapes.py (infer mode)")
    ap.add_argument("--write", action="store_true",
                    help="reparse mode: actually rewrite the CSVs (default is a dry run "
                         "that only reports what would change)")
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
    ap.add_argument("--exemplar-sheet", type=Path, default=None,
                    help="(--mode exemplars) also write a contact sheet PNG of the chosen examples")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit(f"--shard-index must be in [0, {args.num_shards}) for --num-shards {args.num_shards}")

    if args.mode == "exemplars":
        meta = args.few_shot_metadata or args.metadata
        if not meta:
            raise SystemExit("--mode exemplars needs --few-shot-metadata (or --metadata)")
        if args.num_few_shot <= 0:
            raise SystemExit("--mode exemplars needs --num-few-shot >= 1")
        show_exemplars(meta, args.shape_set, args.num_few_shot, args.seed, args.exemplar_sheet)
    elif args.mode == "infer":
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
    elif args.mode == "reparse":
        if not args.results:
            raise SystemExit("--results is required for --mode reparse")
        reparse(args.results, args.shape_set, write=args.write)
    else:
        if not args.results:
            raise SystemExit("--results is required for --mode eval")
        evaluate(args.results)


if __name__ == "__main__":
    main()
