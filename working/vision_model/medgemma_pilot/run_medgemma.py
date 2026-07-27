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
is no server/API backend.

Workflow:
    # 1. Infer per-image (one row per image/plane; resume-safe, re-run to continue):
    python run_medgemma.py --mode infer --metadata meta.csv --out inference_results.csv

    # 2. Aggregate to per-feature majority-vote:
    python run_medgemma.py --mode aggregate --inference-results inference_results.csv \
        --out results_sanity.csv

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

INFERENCE_FIELDS = [
    "case_id", "feature_name", "plane", "modality", "image_path",
    "raw_output", "parsed_label", "ground_truth_label", "correct",
]

RESULT_FIELDS = [
    "case_id", "feature_name", "num_images_used",
    "per_image_labels", "num_images_correct",
    "raw_output", "parsed_label", "ground_truth_label", "correct",
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
    processor = AutoProcessor.from_pretrained(model_id)
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


def run_single(model, processor, messages: List[dict], max_new_tokens: int = 20) -> str:
    """A chat message list -> raw decoded text (greedy).

    `messages` is the backend-neutral format built in prompts.py: a list of
    {"role", "content"} where content items are {"type": "text", ...} or
    {"type": "image", "image": <PIL>}. The processor's chat template inserts the
    image placeholders the way THIS model expects -- do not hand-insert
    <start_of_image>. Few-shot example turns and multi-image prompts are just
    additional messages / image content items; nothing here needs to change.
    """
    import torch

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=model.dtype)  # .to(dtype) casts only float tensors (pixel_values); input_ids stay long

    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.decode(out[0][input_len:], skip_special_tokens=True).strip()


# A backend is just a callable: (messages) -> raw text.
Generate = Callable[[List[dict]], str]


def make_hf_generate(model_id: str, max_new_tokens: int) -> Generate:
    """In-process HuggingFace backend (loads weights locally; needs torch)."""
    model, processor = load_model(model_id)
    return lambda messages: run_single(model, processor, messages, max_new_tokens)


# ---------------------------------------------------------------------------
# Parsing + aggregation
# ---------------------------------------------------------------------------
def parse_answer(raw: str, options: List[str]) -> str:
    """Strict, case-insensitive match to one of `options`; else 'PARSE_FAILED'.
    Prefers an exact one-word answer, then a whole-word occurrence."""
    t = raw.strip().lower()
    lowered = {o.lower(): o for o in options}
    if t in lowered:                      # exact single-word answer
        return lowered[t]
    words = set(t.replace(".", " ").replace(",", " ").split())
    for o_low, o in lowered.items():
        if o_low in words:                # option appears as a standalone word
            return o
    return "PARSE_FAILED"


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


def _overlay_variant(path: Path) -> Optional[Path]:
    """The `_overlay` (red-contour) sibling produced by the preprocessing pipeline."""
    ov = path.with_name(path.stem + "_overlay" + path.suffix)
    return ov if ov.exists() else None


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
) -> None:
    """Infer per-image labels from flattened metadata (one row per image/plane).

    Writes one output row per image; resume-safe (re-running skips images already
    present in out_path). A separate aggregate_results() call majority-votes the
    per-image labels into per-(case, feature) labels.

    num_few_shot > 0 prepends up to that many labeled example turns (from each
    feature's `examples:` block in the YAML) before the query image. Example
    images are held out of inference automatically (leakage guard).
    """
    location_cols = location_cols or ["skeletal_location", "location_within_bone"]
    features = load_feature_config(config_path)
    config_dir = Path(config_path).resolve().parent
    df = pd.read_csv(metadata_csv, dtype=str).fillna("")
    done = _done_keys(out_path)
    log.info("%d image row(s) in metadata; %d already done -> skipping those", len(df), len(done))

    # Few-shot: load example turns per feature once (cached), and collect their
    # image paths so we can exclude them from inference (no train-on-test leakage).
    few_shot_by_feature: Dict[str, List[dict]] = {}
    few_shot_paths: set = set()
    if num_few_shot > 0:
        for feat, fcfg in features.items():
            for p in prompts.few_shot_image_paths(fcfg, config_dir):
                few_shot_paths.add(p)
                few_shot_paths.add(str(Path(p).resolve()))  # match however metadata spells the path
            examples = prompts.resolve_few_shot(fcfg, config_dir, to_jpeg_rgb, limit=num_few_shot)
            few_shot_by_feature[feat] = examples
            log.info("feature %r: %d few-shot example(s) loaded", feat, len(examples))

    # Sibling orientations of the same lesion (assessed in separate calls) -> prompt context.
    planes_by_key: Dict[tuple, List[str]] = {}
    for (cid, feat), g in df.groupby(["case_id", "feature_name"]):
        planes_by_key[(cid, feat)] = [p for p in dict.fromkeys(g["plane"]) if p]

    new_file = not out_path.exists()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=INFERENCE_FIELDS)
        if new_file:
            writer.writeheader()
            fh.flush()

        for _, row in df.iterrows():
            case_id = row["case_id"]
            feature = row["feature_name"]
            img_path_str = row["image_path"]
            if (case_id, feature, img_path_str) in done:
                continue
            if str(Path(img_path_str).resolve()) in few_shot_paths or img_path_str in few_shot_paths:
                log.info("skip %s -- used as a few-shot example (leakage guard)", img_path_str)
                continue
            if feature not in features:
                log.warning("no config for feature %r (case %s) -- skipping", feature, case_id)
                continue
            fcfg = features[feature]

            plane = row.get("plane", "")
            modality = row.get("modality", "")
            img_path = Path(img_path_str)

            has_contour = False
            if use_contour:  # feed the radiologist-contour overlay instead of the plain crop
                ov = _overlay_variant(img_path)
                if ov is not None:
                    img_path, has_contour = ov, True
                else:
                    log.warning("no _overlay for %s -- using plain crop (no contour)", img_path_str)

            try:
                image = to_jpeg_rgb(img_path)
            except Exception as e:  # noqa: BLE001 -- missing/unreadable image
                log.warning("skip image %s (%s / %s): %s", img_path, case_id, feature, e)
                continue

            location = row_location(row, location_cols)
            other_planes = [p for p in planes_by_key.get((case_id, feature), []) if p != plane]
            context = prompts.build_context(
                modality, plane,
                location=location, other_planes=other_planes, has_contour=has_contour,
            )
            messages = prompts.build_medgemma_messages(
                fcfg, image, context, few_shot=few_shot_by_feature.get(feature),
            )
            raw = generate(messages)
            label = parse_answer(raw, fcfg["label_options"])

            gt = row.get("ground_truth_label", "")
            correct = (label.lower() == gt.strip().lower()) if has_gt(gt) and label != "PARSE_FAILED" else ""

            writer.writerow({
                "case_id": case_id,
                "feature_name": feature,
                "plane": plane,
                "modality": modality,
                "image_path": img_path_str,
                "raw_output": raw,
                "parsed_label": label,
                "ground_truth_label": gt,
                "correct": correct,
            })
            fh.flush()
            log.info("%s / %s [%s] -> %s (gt=%s)", case_id, feature, plane or "?", label, gt or "?")

    log.info("done -> %s", out_path)


# ---------------------------------------------------------------------------
# Aggregation (per-image results -> per-feature majority-vote results)
# ---------------------------------------------------------------------------
def aggregate_results(inference_csv: Path, out_path: Path) -> None:
    """Read inference_results.csv (one row per image) and write results_sanity.csv
    with majority-voted labels per (case_id, feature) across all images/planes."""
    df = pd.read_csv(inference_csv, dtype=str).fillna("")
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
        })

    df_agg = pd.DataFrame(aggregated)
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
) -> None:
    """Run a single free-form prompt against one or more images, one image per
    call (same "one image per inference" contract as the rest of the script).
    Prints the raw model output and wall-clock time for each call -- handy for
    a quick runtime/sanity check without needing metadata.csv or the feature
    YAML config.

    `repeat` re-runs the SAME image+prompt N times, useful for timing (e.g.
    measuring steady-state latency after the first, slower, "warm-up" call).
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
            raw = generate(messages)
            dt = time.perf_counter() - t0
            tag = f"{img_path.name}" + (f" (run {i + 1}/{repeat})" if repeat > 1 else "")
            print(f"[{tag}] {dt:.2f}s -> {raw}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", choices=["infer", "aggregate", "eval", "quick"], required=True)
    ap.add_argument("--metadata", type=Path, help="metadata CSV (infer mode)")
    ap.add_argument("--inference-results", type=Path, help="per-image results CSV (aggregate mode)")
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

    if args.mode == "infer":
        if not args.metadata:
            raise SystemExit("--metadata is required for --mode infer")
        generate = make_hf_generate(args.model_id, args.max_new_tokens)
        infer(args.metadata, args.config, args.out or Path("inference_results.csv"), generate,
              use_contour=args.use_contour, location_cols=args.location_cols,
              num_few_shot=args.num_few_shot)
    elif args.mode == "aggregate":
        if not args.inference_results:
            raise SystemExit("--inference-results is required for --mode aggregate")
        aggregate_results(args.inference_results, args.out or args.results)
    elif args.mode == "quick":
        if not args.image or not args.prompt:
            raise SystemExit("--image and --prompt are required for --mode quick")
        generate = make_hf_generate(args.model_id, args.max_new_tokens)
        run_quick(generate, args.image, args.prompt, repeat=args.repeat)
    else:
        evaluate(args.results, args.config)


if __name__ == "__main__":
    main()