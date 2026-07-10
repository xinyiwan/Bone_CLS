"""
Zero-shot bone-tumour feature classification with MedGemma 1.5 (pilot).

ONE IMAGE PER INFERENCE. A feature may list 1-4 images; we run the model once
PER IMAGE and aggregate the per-image labels by majority vote into the final
per-(case, feature) label. This deliberately avoids MedGemma's multi-image path:
per Google's model card, MedGemma's multimodal eval is primarily single-image;
multi-image comprehension is NOT formally evaluated. Single-image + aggregate is
the trustworthy first pass. (A true multi-image variant would go in run_single /
build_prompt -- see the MULTI-IMAGE flag below -- but sanity-check it on a few
known-easy cases before believing it.)

Modes:
    python run_medgemma.py --mode infer --metadata meta.csv --config feature_prompts.yaml --out results.csv
    python run_medgemma.py --mode eval  --results results.csv --config feature_prompts.yaml

LICENSE: MedGemma is governed by the Health AI Developer Foundations (HAI-DEF)
terms of use -- you are responsible for compliance. This script only loads the
public HF weights.

Deps: torch, transformers, pandas, pillow, pyyaml.
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml
from PIL import Image

log = logging.getLogger("medgemma")

RESULT_FIELDS = [
    "case_id", "feature_name", "num_images_used",
    "per_image_labels", "raw_output",
    "parsed_label", "ground_truth_label", "correct",
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
    model_id: str,
    max_new_tokens: int,
    use_contour: bool = False,
    clinical_csv: Optional[Path] = None,
    clinical_key_col: str = "subject",
    location_cols: Optional[List[str]] = None,
) -> None:
    location_cols = location_cols or ["skeletal_location", "location_within_bone"]
    features = load_feature_config(config_path)
    df = pd.read_csv(metadata_csv, dtype=str).fillna("")
    done = _done_keys(out_path)
    log.info("%d row(s) in metadata; %d already done -> skipping those", len(df), len(done))

    clinical = build_clinical_lookup(clinical_csv, clinical_key_col, location_cols) if clinical_csv else {}
    if clinical_csv:
        log.info("loaded anatomical location for %d subject(s)", len(clinical))

    model, processor = load_model(model_id)

    new_file = not out_path.exists()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_FIELDS)
        if new_file:
            writer.writeheader()
            fh.flush()

        for _, row in df.iterrows():
            case_id, feature = row["case_id"], row["feature_name"]
            if (case_id, feature) in done:
                continue
            if feature not in features:
                log.warning("no config for feature %r (case %s) -- skipping", feature, case_id)
                continue
            fcfg = features[feature]

            paths = _split(row["image_paths"])
            planes = _aligned(_split(row.get("plane", "")), len(paths))
            mods = _aligned(_split(row.get("modality", "")), len(paths))
            location = case_location(row, clinical, location_cols)
            distinct_planes = [p for p in dict.fromkeys(planes) if p]  # order-preserving unique

            raws, labels = [], []
            for i, p in enumerate(paths):
                img_path = Path(p)
                has_contour = False
                if use_contour:  # feed the radiologist-contour overlay instead of the plain crop
                    ov = _overlay_variant(img_path)
                    if ov is not None:
                        img_path, has_contour = ov, True
                    else:
                        log.warning("no _overlay for %s -- using plain crop (no contour)", p)
                try:
                    image = to_jpeg_rgb(img_path)
                except Exception as e:  # noqa: BLE001 -- missing/unreadable image
                    log.warning("skip image %s (%s / %s): %s", img_path, case_id, feature, e)
                    continue

                plane_i = planes[i]
                other_planes = [pl for pl in distinct_planes if pl != plane_i]
                prompt = build_prompt(
                    fcfg, mods[i], plane_i,
                    location=location, other_planes=other_planes, has_contour=has_contour,
                )
                raw = run_single(model, processor, image, prompt, max_new_tokens)
                raws.append(raw)
                labels.append(parse_answer(raw, fcfg["label_options"]))

            if not raws:  # nothing usable for this row
                log.warning("no usable images for %s / %s -- skipping row", case_id, feature)
                continue

            final = aggregate(labels)
            gt = row.get("ground_truth_label", "")
            correct = (final.lower() == gt.strip().lower()) if gt and final != "PARSE_FAILED" else ""
            writer.writerow({
                "case_id": case_id,
                "feature_name": feature,
                "num_images_used": len(raws),
                "per_image_labels": ";".join(labels),
                "raw_output": " ||| ".join(raws),
                "parsed_label": final,
                "ground_truth_label": gt,
                "correct": correct,
            })
            fh.flush()
            log.info("%s / %s -> %s (gt=%s, n=%d)", case_id, feature, final, gt or "?", len(raws))

    log.info("done -> %s", out_path)


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
    ap.add_argument("--mode", choices=["infer", "eval"], required=True)
    ap.add_argument("--metadata", type=Path, help="input metadata CSV (infer)")
    ap.add_argument("--results", type=Path, default=Path("results.csv"), help="results CSV (out for infer, in for eval)")
    ap.add_argument("--config", type=Path, default=Path("feature_prompts.yaml"))
    ap.add_argument("--out", type=Path, help="results CSV path for infer (default: --results)")
    ap.add_argument("--model-id", default="google/medgemma-1.5-4b-it",
                    help="default 4B (multi-slice); pass google/medgemma-27b-it for comparison (see load_model note)")
    ap.add_argument("--max-new-tokens", type=int, default=20)
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
        infer(args.metadata, args.config, args.out or args.results, args.model_id, args.max_new_tokens,
              use_contour=args.use_contour, clinical_csv=args.clinical_csv,
              clinical_key_col=args.clinical_key_col, location_cols=args.location_cols)
    else:
        evaluate(args.results, args.config)


if __name__ == "__main__":
    main()
