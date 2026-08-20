"""
Zero-shot bone-tumour feature classification with MedGemma 1.5 (pilot).

ONE IMAGE PER INFERENCE. The preprocessing pipeline outputs one row per
image/plane (no mixing of orientations). We run the model once per image and
output per-image results to inference_results.csv. Then aggregate across images
for the same (case, feature) by majority vote -> results_sanity.csv.

This deliberately avoids MedGemma's multi-image path: per Google's model card,
MedGemma's multimodal eval is primarily single-image; multi-image comprehension
is NOT formally evaluated. Single-image + aggregate is the trustworthy first
pass. (A true multi-image variant would go in run_single -- see the MULTI-IMAGE
flag -- but sanity-check on known-easy cases first.)

The model weights are loaded in-process with transformers (needs torch); there
is no server/API backend. `--backend vllm` swaps the same in-process load for
vLLM (continuous batching) without changing anything else -- both backends
implement the one-line Generate contract (list of message lists -> list of texts).

THROUGHPUT: generation is batched (--batch-size) and shardable across GPUs
(--num-shards / --shard-index). A 4B model decoding one sequence at a time is
memory-bandwidth bound and leaves an A100 nearly idle; see run_batch.

Workflow:
    # 1. Infer per-image (one row per image/plane; resume-safe, re-run to continue):
    python run_medgemma.py --mode infer --metadata meta.csv --out inference_results.csv

    # 1b. Same, but batched (many images per GPU call -- see run_batch) and split
    #     over the node's 4 GPUs (see jobs/run_medgemma_multigpu.sh):
    python run_medgemma.py --mode infer --metadata meta.csv --batch-size 16 \
        --num-shards 4 --shard-index $i --out inference_results.csv   # -> ...shard$i.csv

    # 2. Aggregate to per-feature majority-vote (pass every shard file):
    python run_medgemma.py --mode aggregate --inference-results inference_results.csv \
        --out results_sanity.csv

    # 2b. Merge shards but KEEP one row per image (this is what review_server.py
    #     reads -- the aggregate above drops image_path):
    python run_medgemma.py --mode combine --inference-results inference_results.shard*.csv \
        --out inference_results_all.csv
    python review_server.py --results inference_results_all.csv

    # 3. Eval aggregated results:
    python run_medgemma.py --mode eval --results results_sanity.csv

    # Few-shot infer (prepend N held-out labeled example turns per feature; see feature_prompts.yaml):
    python run_medgemma.py --mode infer --metadata meta.csv --num-few-shot 2 --out inference_fewshot.csv

    # Quick ad-hoc test: one prompt + one or more images, no CSV/YAML needed:
    python run_medgemma.py --mode quick --image scan1.jpg --prompt "Describe this image."

PROMPTS: all prompt construction (system/user message split, few-shot) lives in
prompts.py. This file only orchestrates infer/aggregate/eval.

LICENSE: MedGemma is governed by the Health AI Developer Foundations (HAI-DEF)
terms of use -- you are responsible for compliance. This script only loads the
public HF weights.

Deps: torch, transformers, pandas, pillow, pyyaml.
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd
import yaml
from PIL import Image

import prompts  # prompt construction (see prompts.py)

log = logging.getLogger("medgemma")

# `image_path` is the metadata crop path -- the resume/dedup key (see _done_keys)
# and so NOT necessarily what the model saw. `fed_image_path` is the image
# actually loaded and sent (the `_overlay` sibling under --use-contour), and
# `has_contour` records whether that swap really happened for this row (it can
# fall back to the plain crop when no overlay exists). `model_id`,
# `num_few_shot` and `use_contour` stamp the run config onto every row so a
# results CSV is self-describing.
INFERENCE_FIELDS = [
    "case_id", "feature_name", "plane", "modality", "image_path",
    "fed_image_path", "has_contour", "model_id", "num_few_shot", "use_contour",
    "input_text", "raw_output", "thinking", "parsed_label", "reason",
    "ground_truth_label", "correct",
]

# Run config (model_id / num_few_shot / use_contour) is carried through from the
# per-image rows so the aggregate is self-describing too -- otherwise a
# results_sanity.csv can't say which model or prompt setup produced it.
RESULT_FIELDS = [
    "case_id", "feature_name", "num_images_used",
    "per_image_labels", "num_images_correct",
    "raw_output", "parsed_label", "ground_truth_label", "correct",
    "model_id", "num_few_shot", "use_contour", "num_images_with_contour",
]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_feature_config(path: str | Path) -> Dict[str, dict]:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    return cfg["features"]


# ---------------------------------------------------------------------------
# Image loading  (PNG -> JPEG, in memory)
# ---------------------------------------------------------------------------
def to_jpeg_rgb(path: str | Path, quality: int = 95) -> Image.Image:
    """Load an image and return an RGB PIL image that has been re-encoded as
    JPEG. Gemma's docs recommend JPEG over PNG to avoid encoding-related bias;
    we re-encode in memory (no disk writes) so PNG inputs are handled on the fly.
    Images are already 128x128 crops -- the processor handles the final resize to
    the model's expected input size, so we do NOT resize here."""
    img = Image.open(path).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ---------------------------------------------------------------------------
# Prompt construction lives in prompts.py (per-model strategy + few-shot). This
# module builds MESSAGES (a list of role/content dicts), not a flat string, so
# system/user roles are separated and few-shot examples can be added as prior
# turns. See prompts.build_medgemma_messages / prompts.build_context.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def load_model(model_id: str):
    """Load MedGemma + processor once, in bf16 (80GB GPU -> no quantization).

    NOTE / VERIFY AGAINST THE CURRENT MODEL CARD (details can shift between
    versions):
      - Class: MedGemma 1.5 is a Gemma-3 image-text-to-text model; we load it
        with AutoModelForImageTextToText. Confirm the card doesn't recommend a
        different class or `pipeline(...)` helper.
      - Chat template: we let the PROCESSOR insert image tokens via
        apply_chat_template (see run_single) rather than hand-writing
        <start_of_image>. Confirm the card's message format matches.
      - 27B: pass model_id=google/medgemma-27b-it for a head-to-head. BUT VERIFY
        on HF whether a "1.5"-tagged 27B with the same multi-slice CT/MRI support
        exists -- the multi-slice feature is 1.5-specific. If only the 1.5 4B has
        it, keep 4B as primary and treat 27B as a secondary comparison run.
    """
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = torch.bfloat16  # bf16: stable, and VRAM is not a constraint here
    # use_fast=True -> the fast (torchvision) image processor; silences the
    # "slow image processor" warning and speeds up preprocessing.
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    # LEFT padding is mandatory for BATCHED generation on a decoder-only model:
    # generation continues from the LAST position, so right-padding would make the
    # model continue from pad tokens and emit garbage for every short prompt in the
    # batch. It also makes every sequence's prompt end at the same index, so
    # run_batch can slice off the generated part with a single input_len.
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        # --- smaller-GPU fallback (uncomment + `pip install bitsandbytes`): ---
        # load_in_4bit=True,
        # bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model.eval()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        log.info("GPU memory after load: %.1f GB allocated, %.1f GB reserved", alloc, reserved)
    return model, processor


def run_batch(model, processor, batch: List[List[dict]], max_new_tokens: int = 1024) -> List[str]:
    """A LIST of chat message lists -> one raw decoded text each (greedy).

    Each `messages` is the backend-neutral format built in prompts.py: a list of
    {"role", "content"} where content items are {"type": "text", ...} or
    {"type": "image", "image": <PIL>}. The processor's chat template inserts the
    image placeholders the way THIS model expects -- do not hand-insert
    <start_of_image>. Few-shot example turns and multi-image prompts are just
    additional messages / image content items; nothing here needs to change.

    WHY BATCH: decoding a 4B model at batch size 1 is memory-bandwidth bound --
    every generated token streams all the weights from HBM to serve a single
    sequence, leaving the GPU's compute units almost idle. Batching streams the
    weights once per step for N sequences, so throughput scales nearly linearly
    until the batch becomes compute-bound. Prompts are padded LEFT (see
    load_model), so the prompt of every sequence ends at the same index and
    `out[:, input_len:]` is exactly the generated continuation for each row.

    Caveat inherent to static batching: the call returns only when the SLOWEST
    member of the batch stops, so one long thinking block holds up its whole
    batch. infer() mitigates this by grouping same-feature rows together; the
    vLLM backend removes it entirely via continuous batching.
    """
    import torch

    inputs = processor.apply_chat_template(
        batch,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    ).to(model.device, dtype=model.dtype)  # .to(dtype) casts only float tensors (pixel_values); input_ids stay long

    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return [t.strip() for t in processor.batch_decode(out[:, input_len:], skip_special_tokens=True)]


# A backend is just a callable: (list of message lists) -> list of raw texts.
# Batch-shaped even for a single item, so the HF and vLLM backends are drop-in
# swappable (vLLM only pays off when it is handed many prompts at once).
Generate = Callable[[List[List[dict]]], List[str]]


def make_hf_generate(model_id: str, max_new_tokens: int) -> Generate:
    """In-process HuggingFace backend (loads weights locally; needs torch).

    Retries on CUDA OOM by splitting the batch in half, so an over-large
    --batch-size degrades to slower-but-working instead of killing the run.
    """
    import torch

    model, processor = load_model(model_id)

    def generate(batch: List[List[dict]]) -> List[str]:
        try:
            return run_batch(model, processor, batch, max_new_tokens)
        except torch.cuda.OutOfMemoryError:
            if len(batch) == 1:
                raise
            torch.cuda.empty_cache()
            mid = len(batch) // 2
            log.warning("CUDA OOM at batch size %d -- retrying as %d + %d "
                        "(lower --batch-size to avoid this)", len(batch), mid, len(batch) - mid)
            return generate(batch[:mid]) + generate(batch[mid:])

    return generate


def make_vllm_generate(
    model_id: str,
    max_new_tokens: int,
    max_images_per_prompt: int = 1,
    gpu_memory_utilization: float = 0.90,
    max_model_len: Optional[int] = None,
) -> Generate:
    """vLLM backend: same (messages -> text) contract, much higher throughput.

    vLLM does CONTINUOUS batching -- a finished sequence is immediately replaced
    by a queued one instead of the whole batch waiting on its slowest member --
    plus a paged KV cache and CUDA graphs. Feed it large batches (--batch-size
    256+); the scheduler decides the real per-step batch itself.

    We build the prompt STRING with the HF processor's chat template (tokenize=
    False) and hand the PIL images to vLLM separately via multi_modal_data. That
    is the documented offline multimodal path and it keeps the exact same prompt
    text as the HF backend, so results are comparable between the two.

    VERIFY against the installed vLLM: multimodal Gemma-3 support, and that
    `limit_mm_per_prompt` covers your few-shot image count (1 query image + N
    few-shot images) -- prompts exceeding it are rejected.
    """
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        limit_mm_per_prompt={"image": max_images_per_prompt},
        gpu_memory_utilization=gpu_memory_utilization,
        **({"max_model_len": max_model_len} if max_model_len else {}),
    )
    # do_sample=False in the HF path == greedy == temperature 0 here.
    sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    def generate(batch: List[List[dict]]) -> List[str]:
        requests = []
        for messages in batch:
            text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = [c["image"] for m in messages
                      for c in m.get("content", []) if c.get("type") == "image"]
            req = {"prompt": text}
            if images:
                req["multi_modal_data"] = {"image": images}
            requests.append(req)
        outs = llm.generate(requests, sampling)
        return [o.outputs[0].text.strip() for o in outs]

    return generate


def make_generate(backend: str, model_id: str, max_new_tokens: int, num_few_shot: int = 0) -> Generate:
    """Build the backend named by --backend. Both honour the same Generate contract."""
    if backend == "vllm":
        return make_vllm_generate(model_id, max_new_tokens,
                                  max_images_per_prompt=max(1, num_few_shot + 1))
    return make_hf_generate(model_id, max_new_tokens)


# ---------------------------------------------------------------------------
# Parsing + aggregation
# ---------------------------------------------------------------------------
# MedGemma/Gemma-3 "thinking" mode wraps a chain-of-thought in <unused94>thought
# ... <unused95> BEFORE the final answer. The thought itself echoes the label
# options and draft JSON, so we MUST parse only the answer that follows the last
# end-of-thought marker -- otherwise the label list inside the thought poisons
# both JSON extraction and any word-scan fallback.
THINK_START_MARKERS = ("<unused94>", "<start_of_thought>")
THINK_END_MARKERS = ("<unused95>", "</thought>", "<end_of_turn>")


def _answer_region(raw: str) -> str:
    """Everything after the last end-of-thought marker (the final answer), or the
    whole string when the model didn't emit a thinking block."""
    s = raw
    for marker in THINK_END_MARKERS:
        if marker in s:
            s = s.rsplit(marker, 1)[-1]
    return s


def extract_thinking(raw: str) -> str:
    """The model's full chain-of-thought: the text between the think-start and
    think-end markers. "" when the model emitted no thinking block. This is the
    real reasoning -- richer than the one-line `reason` in the final answer JSON."""
    start = -1
    for m in THINK_START_MARKERS:
        i = raw.find(m)
        if i != -1:
            start = i + len(m)
            break
    if start == -1:
        return ""
    rest = raw[start:].lstrip()
    if rest[:7].lower() == "thought":            # drop the leading "thought" token
        rest = rest[7:].lstrip()
    for m in THINK_END_MARKERS:
        j = rest.find(m)
        if j != -1:
            rest = rest[:j]
            break
    return rest.strip()


def _extract_json(raw: str) -> Optional[dict]:
    """The final JSON object in the answer region, tolerant of ```json fences and
    of the thinking block. Prefers the LAST {...} that parses (the real answer,
    not a draft inside the thought)."""
    import json
    import re

    s = _answer_region(raw).replace("```json", "```")
    if "```" in s:                        # prefer the last fenced chunk with a brace
        for chunk in reversed([c for c in s.split("```") if "{" in c]):
            s = chunk
            break
    s = s.strip()
    try:                                  # clean case: the region is just the JSON
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:  # noqa: BLE001
        pass
    for cand in reversed(re.findall(r"\{[^{}]*\}", s, re.DOTALL)):  # last flat {...}
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:  # noqa: BLE001
            continue
    return None


def parse_answer(raw: str, options: List[str]) -> tuple[str, str]:
    """Parse the model's structured answer into (prediction, reason).

    Reads ONLY the answer region (after the thinking block), most-reliable method
    first:
      1. a well-formed JSON object -> take prediction/reason;
      2. else pull the "prediction"/"reason" keys directly with regex (salvages a
         truncated or loosely-formatted final JSON -- prediction comes first, so
         it survives even when the reason is cut off);
      3. only as a last resort, scan the answer region for a standalone option
         word (never the thought, so the printed options list can't poison it).
    `prediction` is matched case-insensitively to `options` (canonical casing
    returned); 'PARSE_FAILED' when nothing matches. `reason` is "" when absent.
    """
    import re

    lowered = {o.lower(): o for o in options}
    region = _answer_region(raw)
    reason = ""

    # 1. Well-formed JSON.
    obj = _extract_json(raw)
    if obj is not None:
        reason = str(obj.get("reason", "")).strip()
        pred = str(obj.get("prediction", "")).strip().lower()
        if pred in lowered:
            return lowered[pred], reason

    # 2. Loose/truncated JSON: pull the keys straight from the answer region.
    mr = re.search(r'"reason"\s*:\s*"([^"]*)"', region, re.DOTALL)
    if mr and not reason:
        reason = mr.group(1).strip()
    mp = re.search(r'"prediction"\s*:\s*"([^"]*)"', region)
    if mp and mp.group(1).strip().lower() in lowered:
        return lowered[mp.group(1).strip().lower()], reason

    # 3. Last resort: a standalone option word in the answer region.
    words = set(region.lower().replace(".", " ").replace(",", " ").replace('"', " ").split())
    for o_low, o in lowered.items():
        if o_low in words:
            return o, reason
    return "PARSE_FAILED", reason


def has_gt(gt: str) -> bool:
    """A ground-truth label worth scoring: not blank and not the "unknown" marker
    the preprocess step writes for subjects without an assessment label."""
    return str(gt).strip().lower() not in {"", "unknown"}


def aggregate(labels: List[str]) -> str:
    """Majority vote over per-image labels, ignoring PARSE_FAILED. Deterministic
    tie-break: earliest-occurring among the tied labels. All-failed -> PARSE_FAILED."""
    valid = [l for l in labels if l != "PARSE_FAILED"]
    if not valid:
        return "PARSE_FAILED"
    counts = Counter(valid)
    top = max(counts.values())
    tied = {l for l, n in counts.items() if n == top}
    for l in valid:                       # first occurrence wins ties
        if l in tied:
            return l
    return valid[0]


# ---------------------------------------------------------------------------
# Inference loop (resume-safe)
# ---------------------------------------------------------------------------
def _done_keys(out_path: Path) -> set:
    """Already-inferred images, so re-running resumes instead of duplicating.
    Keyed per image: (case_id, feature_name, image_path)."""
    if not out_path.exists():
        return set()
    df = pd.read_csv(out_path, dtype=str).fillna("")
    if df.empty or "image_path" not in df.columns:
        return set()
    return set(zip(df["case_id"], df["feature_name"], df["image_path"]))


def _check_resumable(out_path: Path) -> None:
    """Fail loudly when --out exists but was written with a different column set.

    We append with a fixed DictWriter fieldname list, so appending to a CSV whose
    header doesn't match would silently write values under the wrong headings.
    Older runs (before fed_image_path / model_id / ... were added) hit this."""
    if not out_path.exists() or out_path.stat().st_size == 0:
        return
    with open(out_path, newline="") as fh:
        header = next(csv.reader(fh), [])
    if header != INFERENCE_FIELDS:
        missing = [c for c in INFERENCE_FIELDS if c not in header]
        raise SystemExit(
            f"{out_path} was written with a different schema, so this run cannot append to it.\n"
            f"  missing column(s): {missing or 'none'}\n"
            f"  extra column(s):   {[c for c in header if c not in INFERENCE_FIELDS] or 'none'}\n"
            "Pass a fresh --out path (the old CSV stays valid for aggregate/review)."
        )


def _overlay_variant(path: Path) -> Optional[Path]:
    """The `_overlay` (red-contour) sibling produced by the preprocessing pipeline,
    or None if it doesn't exist. Naming shared with prompts.overlay_variant."""
    ov = prompts.overlay_variant(path)
    return ov if ov.exists() else None


def subject_of_image(path_str: str) -> str:
    """Subject id from a crop path. Preprocess writes crops as
    out_root/<case_id>/<feature>/<file>, and case_id is '<subject>' or, for
    --unit study, '<subject>__<session>' (pipeline._safe_name turns '/' into
    '__'). So the subject is the case_id directory's part before any '__'."""
    return Path(path_str).parent.parent.name.split("__")[0]


def row_location(row: pd.Series, loc_cols: List[str]) -> Optional[str]:
    """Anatomical location for a row, read from metadata columns (the preprocess
    step's --clinical-csv merge writes these in). Blank if none present."""
    parts = [str(row[c]).strip() for c in loc_cols if c in row.index and str(row[c]).strip()]
    return ", ".join(parts) if parts else None


def infer(
    metadata_csv: Path,
    config_path: Path,
    out_path: Path,
    generate: Generate,
    use_contour: bool = False,
    location_cols: Optional[List[str]] = None,
    num_few_shot: int = 0,
    model_id: str = "",
    batch_size: int = 1,
    shard_index: int = 0,
    num_shards: int = 1,
) -> None:
    """Infer per-image labels from flattened metadata (one row per image/plane).

    Writes one output row per image; resume-safe (re-running skips images already
    present in out_path). A separate aggregate_results() call majority-votes the
    per-image labels into per-(case, feature) labels.

    batch_size > 1 sends that many images to the model per generate() call (see
    run_batch for why this is the main throughput lever). Rows are stable-sorted
    by feature first so a batch shares one system prompt -- less padding waste,
    and members finish at more similar lengths, which matters because a static
    batch is only as fast as its slowest member.

    shard_index/num_shards split the work across processes (one per GPU) by
    taking every num_shards-th row. Every shard MUST write its own --out file:
    they append concurrently and would otherwise interleave into corrupt CSV
    rows. Pass all the shard CSVs to --mode aggregate afterwards.

    num_few_shot > 0 prepends up to that many labeled example turns (from each
    feature's `examples:` block in the YAML) before the query image. To avoid
    train-on-test leakage, EVERY image of any subject used as an example is held
    out of inference (subject-level, since an example reveals that subject's
    label); the count is logged at startup.

    `model_id` is recorded on every row (with num_few_shot / use_contour) purely
    so the CSV says which run produced it -- generation itself goes through the
    `generate` callable, which already has the model bound.
    """
    location_cols = location_cols or ["skeletal_location", "location_within_bone"]
    features = load_feature_config(config_path)
    config_dir = Path(config_path).resolve().parent
    df = pd.read_csv(metadata_csv, dtype=str).fillna("")
    # The UNSHARDED frame: the "other planes of this lesion" prompt context must
    # be derived from all images of a case, not just the ones in this shard, or
    # the prompt would silently depend on how the work was split.
    df_full = df
    n_total = len(df)
    if num_shards > 1:
        # Strided (not contiguous) split: every shard gets a mix of cases, so an
        # uneven number of images per case can't leave one GPU running long after
        # the others have finished.
        df = df.iloc[shard_index::num_shards]
        log.info("shard %d/%d -> %d of %d metadata row(s)",
                 shard_index, num_shards, len(df), n_total)
    _check_resumable(out_path)
    done = _done_keys(out_path)
    log.info("%d image row(s) in metadata; %d already done -> skipping those", len(df), len(done))
    log.info("run config: model=%s | %s | contour=%s", model_id or "(unset)",
             f"{num_few_shot}-shot" if num_few_shot > 0 else "zero-shot",
             "on" if use_contour else "off")

    # Few-shot: load example turns per feature once (cached). A few-shot example
    # reveals the label for its whole subject, so to avoid train-on-test leakage
    # we exclude EVERY image of any example subject from inference (not just the
    # exact example image). Subject is read from the crop path's layout
    # (out_root/<case_id>/<feature>/<file>; case_id may be '<subj>__<session>').
    few_shot_by_feature: Dict[str, List[dict]] = {}
    few_shot_subjects: set = set()
    few_shot_paths: set = set()
    if num_few_shot > 0:
        for feat, fcfg in features.items():
            for p in prompts.few_shot_image_paths(fcfg, config_dir):
                few_shot_paths.add(p)
                few_shot_paths.add(str(Path(p).resolve()))
                few_shot_subjects.add(subject_of_image(p))
            examples = prompts.resolve_few_shot(fcfg, config_dir, to_jpeg_rgb,
                                                limit=num_few_shot, use_contour=use_contour)
            few_shot_by_feature[feat] = examples
            log.info("feature %r: %d few-shot example(s) loaded", feat, len(examples))

        # Report how much of the eval set the leakage guard removes.
        subj_series = df_full["case_id"].str.split("/").str[0]
        n_excluded = int(subj_series.isin(few_shot_subjects).sum())
        log.warning("leakage guard: %d example subject(s) %s -> excluding their %d image(s) "
                    "from inference/eval", len(few_shot_subjects), sorted(few_shot_subjects), n_excluded)

    # Sibling orientations of the same lesion (assessed in separate calls) -> prompt context.
    planes_by_key: Dict[tuple, List[str]] = {}
    for (cid, feat), g in df_full.groupby(["case_id", "feature_name"]):
        planes_by_key[(cid, feat)] = [p for p in dict.fromkeys(g["plane"]) if p]

    # --- Pass 1: plan. Resolve every row that still needs inference into a task
    # (all the skip/leakage/contour decisions, and the prompt context), WITHOUT
    # loading images -- so we can order the work before committing memory.
    tasks: List[dict] = []
    for _, row in df.iterrows():
        case_id = row["case_id"]
        feature = row["feature_name"]
        img_path_str = row["image_path"]
        if (case_id, feature, img_path_str) in done:
            continue
        # Leakage guard: skip every image of any subject used as a few-shot
        # example (subject-level, not just the exact example image).
        if case_id.split("/")[0] in few_shot_subjects:
            log.info("skip %s -- subject used as a few-shot example (leakage guard)", img_path_str)
            continue
        if img_path_str in few_shot_paths or str(Path(img_path_str).resolve()) in few_shot_paths:
            log.info("skip %s -- few-shot example image (leakage guard)", img_path_str)
            continue
        if feature not in features:
            log.warning("no config for feature %r (case %s) -- skipping", feature, case_id)
            continue

        plane = row.get("plane", "")
        img_path = Path(img_path_str)

        has_contour = False
        if use_contour:  # feed the radiologist-contour overlay instead of the plain crop
            ov = _overlay_variant(img_path)
            if ov is not None:
                img_path, has_contour = ov, True
            else:
                log.warning("no _overlay for %s -- using plain crop (no contour)", img_path_str)

        other_planes = [p for p in planes_by_key.get((case_id, feature), []) if p != plane]
        tasks.append({
            "case_id": case_id,
            "feature": feature,
            "plane": plane,
            "modality": row.get("modality", ""),
            "image_path": img_path_str,
            "fed_path": img_path,
            "has_contour": has_contour,
            "gt": row.get("ground_truth_label", ""),
            "context": prompts.build_context(
                row.get("modality", ""), plane,
                location=row_location(row, location_cols),
                other_planes=other_planes, has_contour=has_contour,
            ),
        })

    # Group same-feature rows into the same batch: they share the (long) system
    # prompt, so padding waste is minimal and they tend to finish at similar
    # lengths -- which matters because a static batch costs its SLOWEST member.
    # Stable sort, so metadata order is preserved within a feature.
    tasks.sort(key=lambda t: t["feature"])
    n_batches = (len(tasks) + batch_size - 1) // max(batch_size, 1)
    log.info("%d image(s) to infer in %d batch(es) of up to %d", len(tasks), n_batches, batch_size)

    new_file = not out_path.exists()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=INFERENCE_FIELDS)
        if new_file:
            writer.writeheader()
            fh.flush()

        # --- Pass 2: execute, one batch at a time. Images are loaded per batch
        # (not all up front) so peak host memory stays bounded by batch_size.
        t_start = time.perf_counter()
        n_done = 0
        for bi in range(n_batches):
            chunk = tasks[bi * batch_size:(bi + 1) * batch_size]
            batch_messages: List[List[dict]] = []
            batch_tasks: List[dict] = []
            for t in chunk:
                try:
                    image = to_jpeg_rgb(t["fed_path"])
                except Exception as e:  # noqa: BLE001 -- missing/unreadable image
                    log.warning("skip image %s (%s / %s): %s",
                                t["fed_path"], t["case_id"], t["feature"], e)
                    continue
                messages = prompts.build_medgemma_messages(
                    features[t["feature"]], image, t["context"],
                    few_shot=few_shot_by_feature.get(t["feature"]),
                )
                batch_messages.append(messages)
                batch_tasks.append({**t, "input_text": prompts.messages_to_text(messages)})
            if not batch_messages:
                continue

            raws = generate(batch_messages)
            if len(raws) != len(batch_tasks):  # a backend that drops/reorders would misattribute every row
                raise RuntimeError(
                    f"backend returned {len(raws)} output(s) for a batch of {len(batch_tasks)} prompt(s); "
                    "outputs must be one-per-prompt and in order"
                )

            for t, raw in zip(batch_tasks, raws):
                fcfg = features[t["feature"]]
                thinking = extract_thinking(raw)  # full chain-of-thought (richer than `reason`)
                label, reason = parse_answer(raw, fcfg["label_options"])
                if label == "PARSE_FAILED":
                    truncated = ("<unused94>" in raw) and not any(m in raw for m in THINK_END_MARKERS)
                    log.warning("PARSE_FAILED %s / %s%s", t["case_id"], t["feature"],
                                "  (thinking block truncated -- raise --max-new-tokens)" if truncated else "")

                gt = t["gt"]
                # label_options already use the assessment vocabulary, so a direct
                # case-insensitive match is the score (no mapping needed).
                correct = (label.lower() == gt.strip().lower()) if has_gt(gt) and label != "PARSE_FAILED" else ""

                writer.writerow({
                    "case_id": t["case_id"],
                    "feature_name": t["feature"],
                    "plane": t["plane"],
                    "modality": t["modality"],
                    "image_path": t["image_path"],
                    # what the model actually saw -- differs from image_path whenever
                    # the --use-contour overlay swap above succeeded.
                    "fed_image_path": str(t["fed_path"]),
                    "has_contour": t["has_contour"],
                    "model_id": model_id,
                    "num_few_shot": num_few_shot,
                    "use_contour": use_contour,
                    "input_text": t["input_text"],
                    "raw_output": raw,
                    "thinking": thinking,
                    "parsed_label": label,
                    "reason": reason,
                    "ground_truth_label": gt,
                    "correct": correct,
                })
                log.info("%s / %s [%s] -> %s (gt=%s)",
                         t["case_id"], t["feature"], t["plane"] or "?", label, gt or "?")
            # Flush once per batch (not per row): the file stays resume-safe at
            # batch granularity, which is all a killed job can lose.
            fh.flush()

            n_done += len(batch_tasks)
            rate = n_done / max(time.perf_counter() - t_start, 1e-9)
            log.info("batch %d/%d done -- %d/%d image(s), %.2f img/s, ~%.1f min left",
                     bi + 1, n_batches, n_done, len(tasks), rate,
                     (len(tasks) - n_done) / rate / 60 if rate > 0 else float("nan"))

    log.info("done -> %s", out_path)


# ---------------------------------------------------------------------------
# Combining shards / aggregation
# ---------------------------------------------------------------------------
def load_inference_rows(inference_csvs: List[Path] | Path) -> pd.DataFrame:
    """One or more per-image CSVs -> a single de-duplicated per-image frame.

    Shards from a multi-GPU run are disjoint by construction, but we still drop
    duplicate (case, feature, image) rows so an overlapping re-run can't
    double-count a vote."""
    paths = [inference_csvs] if isinstance(inference_csvs, Path) else list(inference_csvs)
    frames = [pd.read_csv(p, dtype=str).fillna("") for p in paths]
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    if len(paths) > 1:
        before = len(df)
        df = df.drop_duplicates(subset=["case_id", "feature_name", "image_path"], keep="first")
        log.info("combined %d shard file(s): %d row(s), %d after de-duplication",
                 len(paths), before, len(df))
    return df


def combine_results(inference_csvs: List[Path] | Path, out_path: Path) -> None:
    """Merge shard CSVs into ONE per-image CSV, keeping every column.

    Unlike aggregate_results (which majority-votes down to one row per
    (case, feature) and so drops image_path / input_text / reason), this keeps
    the per-image rows intact -- which is what review_server.py needs to show
    each image next to its prediction. Use this on a sharded multi-GPU run
    before opening the viewer."""
    df = load_inference_rows(inference_csvs)
    if df.empty:
        log.warning("inference results CSV is empty")
        return
    # Preserve the canonical column order; keep any extra columns at the end so
    # nothing is silently dropped from an older or hand-edited CSV.
    ordered = [c for c in INFERENCE_FIELDS if c in df.columns]
    df = df[ordered + [c for c in df.columns if c not in ordered]]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info("combined %d per-image row(s) -> %s", len(df), out_path)
    if "image_path" not in df.columns:
        log.warning("no image_path column -- review_server.py will not be able to show images")


def aggregate_results(inference_csvs: List[Path] | Path, out_path: Path) -> None:
    """Read one or more inference_results CSVs (one row per image) and write
    results_sanity.csv with majority-voted labels per (case_id, feature) across
    all images/planes.

    Accepts several inputs so a sharded multi-GPU run (one CSV per GPU) can be
    aggregated directly -- the shards are disjoint by construction, but we still
    drop duplicate (case, feature, image) rows so a re-run overlap can't
    double-count a vote."""
    df = load_inference_rows(inference_csvs)
    if df.empty:
        log.warning("inference results CSV is empty")
        return

    aggregated = []
    for (case_id, feature), group in df.groupby(["case_id", "feature_name"]):
        labels = group["parsed_label"].tolist()
        final = aggregate(labels)
        gt = group["ground_truth_label"].iloc[0]  # assume all images for a (case, feature) have the same gt
        correct = (final.lower() == gt.strip().lower()) if has_gt(gt) and final != "PARSE_FAILED" else ""

        # Per-image labels and raw outputs, one per image in the group.
        per_img = ";".join(group["parsed_label"].tolist())
        num_correct = sum(1 for l in labels if l.lower() == gt.strip().lower()) if has_gt(gt) else 0
        raws = " ||| ".join(group["raw_output"].tolist())

        # Run config carried through from the per-image rows, so the aggregate
        # says which model / prompt setup produced it. Joined over the distinct
        # values in the group, so a file mixing runs shows both rather than
        # silently reporting the first. Blank for pre-schema inputs.
        def cfg(col: str) -> str:
            if col not in group.columns:
                return ""
            return ";".join(sorted({str(v).strip() for v in group[col] if str(v).strip()}))

        aggregated.append({
            "case_id": case_id,
            "feature_name": feature,
            "num_images_used": len(group),
            "per_image_labels": per_img,
            "num_images_correct": num_correct,
            "raw_output": raws,
            "parsed_label": final,
            "ground_truth_label": gt,
            "correct": correct,
            "model_id": cfg("model_id"),
            "num_few_shot": cfg("num_few_shot"),
            "use_contour": cfg("use_contour"),
            # How many of this group's images really carried a contour overlay
            # (can be < num_images_used when an _overlay was missing).
            "num_images_with_contour": sum(
                1 for v in group.get("has_contour", []) if str(v).strip().lower() in {"true", "1"}
            ),
        })

    df_agg = pd.DataFrame(aggregated, columns=RESULT_FIELDS)
    df_agg.to_csv(out_path, index=False)
    log.info("aggregated %d (case, feature) pairs -> %s", len(aggregated), out_path)


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------
def evaluate(results_csv: Path, config_path: Optional[Path]) -> None:
    df = pd.read_csv(results_csv, dtype=str).fillna("")
    scored = df[df["ground_truth_label"].map(has_gt)].copy()
    if scored.empty:
        print("No rows with a known ground_truth_label -- nothing to score.")
        return
    scored["is_correct"] = scored["correct"].astype(str).str.lower().isin({"true", "1"})
    scored["nimg"] = pd.to_numeric(scored["num_images_used"], errors="coerce").fillna(0).astype(int)

    def bucket(n: int) -> str:
        return "1" if n == 1 else "2" if n == 2 else "3-4" if n >= 3 else "0"

    print(f"\n=== Overall ===  n={len(scored)}  accuracy={scored['is_correct'].mean():.3f}"
          f"  parse_failures={(scored['parsed_label'] == 'PARSE_FAILED').sum()}")

    for feat, g in scored.groupby("feature_name"):
        print(f"\n=== {feat} ===  n={len(g)}  accuracy={g['is_correct'].mean():.3f}"
              f"  parse_failures={(g['parsed_label'] == 'PARSE_FAILED').sum()}")
        cm = pd.crosstab(g["ground_truth_label"], g["parsed_label"])
        print("confusion (rows=truth, cols=pred):")
        print(cm.to_string())

    # Multi-image diagnostic: does aggregating more views help or hurt?
    print("\n=== Accuracy by num_images_used (aggregation diagnostic) ===")
    scored["bucket"] = scored["nimg"].map(bucket)
    by = scored.groupby("bucket")["is_correct"].agg(["count", "mean"]).rename(columns={"mean": "accuracy"})
    print(by.to_string())


# ---------------------------------------------------------------------------
# Quick ad-hoc test  (no CSV / YAML needed -- just a prompt + image(s))
# ---------------------------------------------------------------------------
def run_quick(
    generate: Generate,
    image_paths: List[Path],
    prompt: str,
    repeat: int = 1,
    batch_size: int = 1,
) -> None:
    """Run a single free-form prompt against one or more images, one image per
    call (same "one image per inference" contract as the rest of the script).
    Prints the raw model output and wall-clock time for each call -- handy for
    a quick runtime/sanity check without needing metadata.csv or the feature
    YAML config.

    `repeat` re-runs the SAME image+prompt N times, useful for timing (e.g.
    measuring steady-state latency after the first, slower, "warm-up" call).
    `batch_size` sends that many copies of the prompt in one call -- the quickest
    way to measure the throughput gain from batching before committing to a
    --batch-size for a real infer run. Always discard the first (warm-up) run.
    """
    for img_path in image_paths:
        try:
            image = to_jpeg_rgb(img_path)
        except Exception as e:  # noqa: BLE001 -- missing/unreadable image
            print(f"[{img_path}] could not load image: {e}")
            continue

        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
        }]
        for i in range(repeat):
            t0 = time.perf_counter()
            # batch of `batch_size` copies of the same prompt: with batch_size > 1
            # this measures per-image throughput at that batch size, which is the
            # number to compare against batch 1 when picking --batch-size.
            raws = generate([messages] * batch_size)
            dt = time.perf_counter() - t0
            tag = f"{img_path.name}" + (f" (run {i + 1}/{repeat})" if repeat > 1 else "")
            if batch_size > 1:
                print(f"[{tag}] batch={batch_size}  {dt:.2f}s total, "
                      f"{dt / batch_size:.2f}s/image, {batch_size / dt:.2f} img/s -> {raws[0]}")
            else:
                print(f"[{tag}] {dt:.2f}s -> {raws[0]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", choices=["infer", "combine", "aggregate", "eval", "quick"], required=True)
    ap.add_argument("--metadata", type=Path, help="metadata CSV (infer mode)")
    ap.add_argument("--inference-results", type=Path, nargs="+",
                    help="per-image results CSV(s) (combine/aggregate mode); pass every shard file "
                         "from a multi-GPU run and they are merged first")
    ap.add_argument("--results", type=Path, default=Path("results_sanity.csv"),
                    help="results CSV (eval mode input, or aggregate mode output)")
    ap.add_argument("--config", type=Path, default=Path("feature_prompts.yaml"),
                    help="feature config YAML")
    ap.add_argument("--out", type=Path,
                    help="output CSV (infer: default inference_results.csv; aggregate: default --results)")
    # model / decoding
    ap.add_argument("--model-id", default="google/medgemma-1.5-4b-it",
                    help="default 4B; pass google/medgemma-27b-it for a comparison run (see load_model)")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    # throughput
    ap.add_argument("--backend", choices=["hf", "vllm"], default="hf",
                    help="hf: in-process transformers with static batching (no extra deps). "
                         "vllm: continuous batching + paged KV cache, faster but needs vllm installed "
                         "and a large --batch-size to pay off")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="images per generate() call. 1 = the old row-by-row behaviour. "
                         "8-32 is a good range for the 4B on an 80GB A100 (hf); 256+ for vllm. "
                         "Measure first with --mode quick --batch-size N --repeat 3")
    ap.add_argument("--shard-index", type=int, default=0,
                    help="this process's shard (0-based); see --num-shards")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="split the metadata across N processes (one per GPU) for data-parallel "
                         "inference. Each shard MUST get its own --out; aggregate all shard CSVs "
                         "afterwards with --mode aggregate --inference-results <all shards>")
    # prompt context
    ap.add_argument("--use-contour", action="store_true",
                    help="feed the radiologist red-contour '_overlay' image and tell the model about it")
    ap.add_argument("--location-cols", nargs="*", default=["skeletal_location", "location_within_bone"],
                    help="metadata columns joined into the lesion-location phrase (added by preprocess --clinical-csv)")
    ap.add_argument("--num-few-shot", type=int, default=0,
                    help="prepend up to N labeled example turns per feature from the YAML 'examples:' block "
                         "(0 = zero-shot; example images are auto-excluded from inference)")
    # quick mode
    ap.add_argument("--image", type=Path, nargs="+",
                    help="one or more image paths (quick mode); each is sent in its own call")
    ap.add_argument("--prompt", type=str, help="free-form text prompt (quick mode)")
    ap.add_argument("--repeat", type=int, default=1,
                    help="repeat the same image+prompt N times (quick mode; useful for timing runs)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit(f"--shard-index must be in [0, {args.num_shards}) for --num-shards {args.num_shards}")

    if args.mode == "infer":
        if not args.metadata:
            raise SystemExit("--metadata is required for --mode infer")
        out = args.out or Path("inference_results.csv")
        if args.num_shards > 1:
            # Shards append concurrently; sharing one --out would interleave
            # half-written rows. Give each its own file, aggregate afterwards.
            out = out.with_name(f"{out.stem}.shard{args.shard_index}{out.suffix}")
            log.info("sharded run -> writing %s", out)
        generate = make_generate(args.backend, args.model_id, args.max_new_tokens, args.num_few_shot)
        infer(args.metadata, args.config, out, generate,
              use_contour=args.use_contour, location_cols=args.location_cols,
              num_few_shot=args.num_few_shot, model_id=args.model_id,
              batch_size=args.batch_size,
              shard_index=args.shard_index, num_shards=args.num_shards)
    elif args.mode == "combine":
        if not args.inference_results:
            raise SystemExit("--inference-results is required for --mode combine")
        if not args.out:
            raise SystemExit("--out is required for --mode combine (the merged per-image CSV)")
        combine_results(args.inference_results, args.out)
    elif args.mode == "aggregate":
        if not args.inference_results:
            raise SystemExit("--inference-results is required for --mode aggregate")
        aggregate_results(args.inference_results, args.out or args.results)
    elif args.mode == "quick":
        if not args.image or not args.prompt:
            raise SystemExit("--image and --prompt are required for --mode quick")
        generate = make_generate(args.backend, args.model_id, args.max_new_tokens, args.num_few_shot)
        run_quick(generate, args.image, args.prompt, repeat=args.repeat, batch_size=args.batch_size)
    else:
        evaluate(args.results, args.config)


if __name__ == "__main__":
    main()