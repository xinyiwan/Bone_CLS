# MedGemma feature classifier (pilot)

Lean evaluation of `google/medgemma-1.5-4b-it` on preprocessed 2D bone-tumour MRI
crops. Zero-shot by default; optional few-shot via held-out labeled exemplars.

**Approach:** one image per inference call. The preprocessing pipeline outputs
one row per image/plane (flattened — no mixing of orientations). We infer each
image independently and write `inference_results.csv` (per-image labels), then
aggregate across images for each feature via majority vote into
`results_sanity.csv` (per-feature labels). This avoids MedGemma's unvalidated
multi-image path (see the note atop `run_medgemma.py`).

## Install

Model weights load in-process with transformers (no server/API backend):

```bash
pip install torch transformers pandas pillow pyyaml
```

Or in a container: `docker build -f Dockerfile.hf -t medgemma-hf --build-arg HF_TOKEN=hf_xxx .`

## Input

**Metadata CSV** from the preprocessing pipeline:
`case_id, feature_name, plane, modality, slice_index, image_path, crop_bbox, margin_used, ground_truth_label`

One row = one image (flattened: no mixing of orientations). The pipeline
automatically reads this and processes each image independently.

`ground_truth_label` is filled by the preprocess step from each subject's
assessment JSON (`run.py --labels-dir ...`). Subjects without a label get
`unknown` — those rows are still inferred, but **skipped when scoring**
(`eval` and the `correct` column treat `unknown` like a blank).

**Feature config** (`feature_prompts.yaml`):
Feature vocab + prompt wording — not hardcoded. Prompt assembly lives in
`prompts.py`, which builds chat **messages** (not a flat string): one system
message carries the constant task (role + each feature's `description` →
`label_definitions` (optional) → `task` → strict answer format from
`label_options`), and each user turn carries the image + its per-image context
(modality/plane/location/contour, built by `prompts.build_context`). To tune
wording for a feature, edit the YAML; to change the framing for *all* features,
edit `prompts.SYSTEM_ROLE` / `prompts.build_system_text` / `prompts.build_context`.

**Few-shot (optional):** add an `examples:` block to a feature in the YAML (see
the commented templates there) listing held-out labeled images, then pass
`--num-few-shot N`. Each example becomes a prior (user image → assistant label)
turn before the query image; example images are auto-excluded from inference to
avoid leakage. Default (`--num-few-shot 0`) is zero-shot.

## Workflow

```bash
# 1. Infer per-image (one row per image/plane):
python run_medgemma.py --mode infer \
    --metadata ../preprocess/metadata.csv --config feature_prompts.yaml \
    --out inference_results.csv

# 2. Aggregate to per-feature (majority-vote across images for each case+feature):
python run_medgemma.py --mode aggregate \
    --inference-results inference_results.csv --out results_sanity.csv

# 3. Eval aggregated results:
python run_medgemma.py --mode eval --results results_sanity.csv --config feature_prompts.yaml
```

**Few-shot run** (after adding `examples:` to the YAML — see "Few-shot" above):
```bash
python run_medgemma.py --mode infer --num-few-shot 2 \
    --metadata ../preprocess/metadata.csv --config feature_prompts.yaml \
    --out inference_fewshot.csv
```

**Sanity check first** (optional, on a few known-easy cases):
```bash
# Subset your metadata to easy cases, then run steps 1–3 above
python run_medgemma.py --mode infer \
    --metadata easy_subset.csv --config feature_prompts.yaml \
    --out inference_results_sanity.csv
python run_medgemma.py --mode aggregate \
    --inference-results inference_results_sanity.csv --out results_sanity_sanity.csv
```

**Comparison run:**
Pass `--model-id google/medgemma-27b-it` to use the 27B model instead of 4B
(verify a 1.5-tagged 27B exists on the hub first — see `load_model` note).

## Outputs

**`inference_results.csv`** (from `--mode infer`):
One row per image processed.
Columns: `case_id, feature_name, plane, modality, image_path, raw_output, parsed_label, ground_truth_label, correct`

Use this to inspect per-image predictions and debug which images disagreed.

**`results_sanity.csv`** (from `--mode aggregate`):
One row per (case_id, feature) with majority-voted label.
Columns: `case_id, feature_name, num_images_used, per_image_labels, num_images_correct, raw_output, parsed_label, ground_truth_label, correct`

- `per_image_labels` — semicolon-separated per-image predictions (for seeing the vote)
- `num_images_correct` — how many images voted for the final label (if gt is known)

**Eval output** (from `--mode eval`):
Prints overall + per-feature accuracy, confusion matrices, and **accuracy by
`num_images_used`** — the diagnostic for whether aggregating more views helps or
hurts.

## Before trusting results

- **Sanity-check the model** on a handful of obvious cases — MedGemma's card
  notes multimodal eval is primarily single-image.
- **Verify the API** against the current model card: model class, chat-template
  message format, and JPEG recommendation are flagged inline in `run_medgemma.py`
  where I relied on Gemma-3 conventions that may shift between versions.
- MedGemma use is governed by the **HAI-DEF** license terms.
