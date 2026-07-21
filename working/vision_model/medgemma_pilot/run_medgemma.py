"""
Zero-shot bone-tumour feature classification with MedGemma 1.5 (pilot).

ONE IMAGE PER INFERENCE. The preprocessing pipeline outputs one row per
image/plane (no mixing of orientations). We run the model once per image and
output per-image results to inference_results.csv. Then aggregate across images
for the same (case, feature) by majority vote -> results_sanity.csv.

This deliberately avoids MedGemma's multi-image path: per Google's model card,
MedGemma's multimodal eval is primarily single-image; multi-image comprehension
is NOT formally evaluated. Single-image + aggregate is the trustworthy first
pass. (A true multi-image variant would go in run_single / build_prompt -- see
the MULTI-IMAGE flag -- but sanity-check on known-easy cases first.)

Two backends (same code path otherwise):
  - hf     : load the weights in-process (needs torch/transformers).
  - openai : call a locally-served OpenAI-compatible endpoint (vllm serve).
             Client needs only openai + pandas + pillow + pyyaml, no torch.

Workflow:
    # 1. Infer per-image (one row per image/plane):
    python run_medgemma.py --mode infer --metadata meta.csv \
        --config feature_prompts.yaml --out inference_results.csv

    # 2. Aggregate to per-feature majority-vote:
    python run_medgemma.py --mode aggregate --inference-results inference_results.csv \
        --out results_sanity.csv

    # 3. Eval aggregated results:
    python run_medgemma.py --mode eval --results results_sanity.csv --config feature_prompts.yaml

LICENSE: MedGemma is governed by the Health AI Developer Foundations (HAI-DEF)
terms of use -- you are responsible for compliance. This script only loads the
public HF weights.

Deps: torch, transformers, pandas, pillow, pyyaml.
"""

from __future__ import annotations

import argparse
import base64
import csv
import logging
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd
import yaml
from PIL import Image

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
# Prompt
# ---------------------------------------------------------------------------
def build_prompt(
    feature_cfg: dict,
    modality: str,
    plane: str,
    location: Optional[str] = None,
    other_planes: Optional[List[str]] = None,
    has_contour: bool = False,
) -> str:
    """Assemble a single-image prompt in a fixed STRUCTURE. All feature-specific
    wording comes from feature_cfg (the YAML); this function only supplies the
    general imaging/clinical context and the strict answer format.

        [general context]  <- here (modality/plane/location/contour/other views)
        + description       <- feature_cfg["description"]
        + label definitions <- feature_cfg["label_definitions"] (optional)
        + task              <- feature_cfg["task"]
        + answer format     <- here (from feature_cfg["label_options"])

    Args:
        modality:      e.g. "T1", "T2FS", "T1FSC"  (per THIS image)
        plane:         orientation of THIS image (axial/coronal/sagittal)
        location:      lesion location, e.g. "distal femur, metaphysis" (from clinical CSV; optional)
        other_planes:  other orientations of the SAME lesion assessed in separate calls,
                       so the model knows this image is one view of a multi-view protocol.
        has_contour:   True when a radiologist red-contour overlay is being fed.

    NOTE (single-image protocol): we assess ONE image per call and aggregate, so
    "other orientations are assessed separately" is accurate. If you switch to
    several images in ONE prompt (MULTI-IMAGE FLAG in run_single), change that
    wording to "shown alongside" and re-validate.
    """
    modality = (modality or "").strip() or "MRI"
    plane = (plane or "").strip() or "unknown-plane"
    opts = ", ".join(feature_cfg["label_options"])

    parts: List[str] = []

    # --- general context (structural; built from runtime values) ---
    loc = f", from a bone lesion located in the {location}" if location else ", from a bone lesion"
    parts.append(
        f"You are an expert musculoskeletal radiologist. You are shown a {modality} MRI "
        f"image in the {plane} plane{loc}. The image is cropped to a bounding box around "
        f"the lesion (with a small margin), so the lesion fills most of the frame."
    )
    parts.append(
        f"This is the {plane} slice with the LARGEST cross-sectional area of the lesion; "
        f"make your assessment based on this slice."
    )
    if other_planes:
        others = ", ".join(other_planes)
        parts.append(
            f"To help build a 3D picture of the same lesion, its largest-area slice in "
            f"other orientations ({others}) is assessed separately in other images; "
            f"judge only the image shown here."
        )
    if has_contour:
        parts.append(
            "A thin RED contour drawn on the image marks the lesion boundary segmented by "
            "a radiologist. Use it to locate the lesion; assess the region it encloses."
        )

    # --- feature-specific wording (from config; description kept for back-compat) ---
    description = (feature_cfg.get("description") or feature_cfg.get("prompt_description") or "").strip()
    if description:
        parts.append(description)
    defs = feature_cfg.get("label_definitions")
    if defs:
        parts.append("Label meanings -- " + "; ".join(f"{k}: {v}" for k, v in defs.items()) + ".")
    task = (feature_cfg.get("task") or "").strip()
    if task:
        parts.append(task)

    # --- strict answer format (structural; guarantees parse_answer alignment) ---
    parts.append(f"Respond with exactly one word from: {opts}.")
    return " ".join(parts)


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


def run_single(model, processor, image: Image.Image, prompt: str, max_new_tokens: int = 20) -> str:
    """One image + one prompt -> raw decoded text (greedy).

    Uses the processor's chat template so image placeholders are inserted the way
    THIS model expects -- do not hand-insert <start_of_image>.

    MULTI-IMAGE FLAG: to feed several images in ONE prompt later, add more
    {"type": "image", ...} entries to `content` (Gemma 3 supports repeated image
    tokens; all images should be the same shape and a batch pads to a fixed
    count). Not used here -- sanity-check before trusting it.
    """
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
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


def run_single_openai(client, model_id: str, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
    """One image + one prompt against an OpenAI-compatible server (e.g. a local
    `vllm serve`). The image is sent as a base64 JPEG data URL.

    VERIFY: multi-modal chat via vllm needs a vllm version with Gemma-3/MedGemma
    vision support, and the server may need `--limit-mm-per-prompt image=1`.
    We send text + one image_url; confirm the served model's chat template
    accepts this content format.
    """
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=95)
    b64 = base64.b64encode(buf.getvalue()).decode()
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        max_tokens=max_new_tokens,
        temperature=0.0,  # greedy, deterministic short-label task
    )
    return (resp.choices[0].message.content or "").strip()


# A backend is just a callable: (image, prompt) -> raw text.
Generate = Callable[[Image.Image, str], str]


def make_hf_generate(model_id: str, max_new_tokens: int) -> Generate:
    """In-process HuggingFace backend (loads weights locally; needs torch)."""
    model, processor = load_model(model_id)
    return lambda image, prompt: run_single(model, processor, image, prompt, max_new_tokens)


def make_openai_generate(base_url: str, api_key: str, model_id: str, max_new_tokens: int) -> Generate:
    """Client backend for a locally-served OpenAI-compatible endpoint (vllm).
    No torch/transformers needed on this side."""
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY")
    log.info("using OpenAI-compatible backend at %s (model=%s)", base_url, model_id)
    return lambda image, prompt: run_single_openai(client, model_id, image, prompt, max_new_tokens)


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
    if not out_path.exists():
        return set()
    df = pd.read_csv(out_path, dtype=str)
    return set(zip(df["case_id"], df["feature_name"]))


def _split(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(";") if x.strip()]


def _aligned(values: List[str], n: int, default: str = "") -> List[str]:
    """Make a ';'-parallel column line up with the n images. If it already has n
    entries use them; if it has exactly one, broadcast it; otherwise pad/truncate."""
    if len(values) == n:
        return values
    if len(values) == 1:
        return values * n
    return (values + [default] * n)[:n]


def _overlay_variant(path: Path) -> Optional[Path]:
    """The `_overlay` (red-contour) sibling produced by the preprocessing pipeline."""
    ov = path.with_name(path.stem + "_overlay" + path.suffix)
    return ov if ov.exists() else None


def build_clinical_lookup(clinical_csv: Path, key_col: str, loc_cols: List[str]) -> Dict[str, str]:
    """subject -> "skeletal_location, location_within_bone" (blanks dropped)."""
    df = pd.read_csv(clinical_csv, dtype=str).fillna("")
    if key_col not in df.columns:
        raise SystemExit(f"{clinical_csv}: no key column {key_col!r} (have {list(df.columns)})")
    present = [c for c in loc_cols if c in df.columns]
    if not present:
        log.warning("clinical CSV has none of the location columns %s", loc_cols)
    out: Dict[str, str] = {}
    for _, r in df.iterrows():
        parts = [r[c].strip() for c in present if r[c].strip()]
        if parts:
            out[r[key_col].strip()] = ", ".join(parts)
    return out


def case_location(row: pd.Series, clinical: Dict[str, str], loc_cols: List[str]) -> Optional[str]:
    """Anatomical location for a row: prefer columns already in the metadata,
    else fall back to the clinical lookup keyed by subject (case_id before '/')."""
    parts = [str(row[c]).strip() for c in loc_cols if c in row.index and str(row[c]).strip()]
    if parts:
        return ", ".join(parts)
    subject = str(row["case_id"]).split("/")[0]
    return clinical.get(subject)


def infer(
    metadata_csv: Path,
    config_path: Path,
    out_path: Path,
    generate: Generate,
    use_contour: bool = False,
    clinical_csv: Optional[Path] = None,
    clinical_key_col: str = "subject",
    location_cols: Optional[List[str]] = None,
) -> None:
    """Infer per-image labels from flattened metadata (one row per image/plane).

    Outputs inference_results.csv with one row per image processed. A separate
    aggregate_results() call produces results_sanity.csv with majority-voted
    labels per (case_id, feature).
    """
    location_cols = location_cols or ["skeletal_location", "location_within_bone"]
    features = load_feature_config(config_path)
    df = pd.read_csv(metadata_csv, dtype=str).fillna("")
    log.info("%d row(s) in metadata (one per image/plane)", len(df))

    clinical = build_clinical_lookup(clinical_csv, clinical_key_col, location_cols) if clinical_csv else {}
    if clinical_csv:
        log.info("loaded anatomical location for %d subject(s)", len(clinical))

    new_file = not out_path.exists()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=INFERENCE_FIELDS)
        if new_file:
            writer.writeheader()
            fh.flush()

        for idx, row in df.iterrows():
            case_id = row["case_id"]
            feature = row["feature_name"]
            if feature not in features:
                log.warning("no config for feature %r (case %s) -- skipping", feature, case_id)
                continue
            fcfg = features[feature]

            img_path = Path(row["image_path"])
            plane = row.get("plane", "")
            modality = row.get("modality", "")

            has_contour = False
            if use_contour:
                ov = _overlay_variant(img_path)
                if ov is not None:
                    img_path, has_contour = ov, True

            try:
                image = to_jpeg_rgb(img_path)
            except Exception as e:  # noqa: BLE001
                log.warning("skip image %s (%s / %s): %s", img_path, case_id, feature, e)
                continue

            location = case_location(row, clinical, location_cols)
            prompt = build_prompt(fcfg, modality, plane, location=location, has_contour=has_contour)
            raw = generate(image, prompt)
            label = parse_answer(raw, fcfg["label_options"])

            gt = row.get("ground_truth_label", "")
            correct = (label.lower() == gt.strip().lower()) if gt and label != "PARSE_FAILED" else ""

            writer.writerow({
                "case_id": case_id,
                "feature_name": feature,
                "plane": plane,
                "modality": modality,
                "image_path": row["image_path"],
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
        correct = (final.lower() == gt.strip().lower()) if gt and final != "PARSE_FAILED" else ""

        # Per-image labels and raw outputs, one per image in the group.
        per_img = ";".join(group["parsed_label"].tolist())
        num_correct = sum(1 for l in labels if l.lower() == gt.strip().lower()) if gt and gt != "PARSE_FAILED" else 0
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
    scored = df[df["ground_truth_label"] != ""].copy()
    if scored.empty:
        print("No rows with ground_truth_label -- nothing to score.")
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
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", choices=["infer", "aggregate", "eval"], required=True)
    ap.add_argument("--metadata", type=Path, help="input metadata CSV (infer)")
    ap.add_argument("--inference-results", type=Path, help="per-image results CSV (aggregate)")
    ap.add_argument("--results", type=Path, default=Path("results_sanity.csv"),
                    help="results CSV (in for eval, out for aggregate)")
    ap.add_argument("--config", type=Path, default=Path("feature_prompts.yaml"))
    ap.add_argument("--out", type=Path, help="output CSV path (infer/aggregate; default: inference_results.csv / results_sanity.csv)")
    ap.add_argument("--model-id", default="google/medgemma-1.5-4b-it",
                    help="default 4B (multi-slice); pass google/medgemma-27b-it for comparison")
    ap.add_argument("--max-new-tokens", type=int, default=20)
    # backend: in-process HF weights, or a locally-served OpenAI-compatible endpoint (vllm)
    ap.add_argument("--backend", choices=["hf", "openai"], default="hf",
                    help="hf = load weights in-process (needs torch); openai = call a local vllm server")
    ap.add_argument("--base-url", default="http://localhost:8000/v1", help="OpenAI-compatible server URL (openai backend)")
    ap.add_argument("--api-key", default="EMPTY", help="API key for the server (vllm ignores it; any non-empty string)")
    # context enrichment
    ap.add_argument("--use-contour", action="store_true",
                    help="feed the radiologist red-contour '_overlay' image and tell the model about it")
    ap.add_argument("--clinical-csv", type=Path,
                    help="per-subject clinical info (e.g. combine_cli_info.py output) for anatomical location")
    ap.add_argument("--clinical-key-col", default="subject", help="subject-id column in the clinical CSV")
    ap.add_argument("--location-cols", nargs="*", default=["skeletal_location", "location_within_bone"],
                    help="clinical columns joined into the lesion location phrase")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.mode == "infer":
        if not args.metadata:
            raise SystemExit("--metadata is required for --mode infer")
        if args.backend == "openai":
            generate = make_openai_generate(args.base_url, args.api_key, args.model_id, args.max_new_tokens)
        else:
            generate = make_hf_generate(args.model_id, args.max_new_tokens)
        infer(args.metadata, args.config, args.out or Path("inference_results.csv"), generate,
              use_contour=args.use_contour, clinical_csv=args.clinical_csv,
              clinical_key_col=args.clinical_key_col, location_cols=args.location_cols)
    elif args.mode == "aggregate":
        if not args.inference_results:
            raise SystemExit("--inference-results is required for --mode aggregate")
        aggregate_results(args.inference_results, args.out or args.results)
    else:
        evaluate(args.results, args.config)


if __name__ == "__main__":
    main()
