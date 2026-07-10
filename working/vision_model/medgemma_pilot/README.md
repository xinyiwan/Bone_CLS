# MedGemma zero-shot feature classifier (pilot)

Lean zero-shot evaluation of `google/medgemma-1.5-4b-it` on preprocessed 2D bone-
tumour MRI crops. **No few-shot, no logprobs** — a correct end-to-end run first.

**Approach:** one image per inference call. For a feature with several images we
run the model once per image and **majority-vote** the per-image labels. This
avoids MedGemma's unvalidated multi-image path (see the note atop `run_medgemma.py`).

## Two ways to run

**A. Local vLLM server (recommended, matches your Docker workflow).** The model
runs in a container behind an OpenAI-compatible API; the script is a thin client
(no torch needed on the client).

```bash
# build + serve MedGemma (needs NVIDIA driver + nvidia-container-toolkit on host)
docker build -t medgemma-serve --build-arg HF_TOKEN=hf_xxx .
docker run --gpus all -p 8000:8000 medgemma-serve      # serves on :8000

# client (only needs: openai pandas pillow pyyaml)
pip install --user -r requirements.txt
python run_medgemma.py --mode infer --backend openai \
    --base-url http://localhost:8000/v1 --model-id google/medgemma-1.5-4b-it \
    --metadata meta.csv --config feature_prompts.yaml --out results.csv
```

**B. In-process (no server).** Loads the weights directly; needs torch.

```bash
pip install --user torch transformers pandas pillow pyyaml
python run_medgemma.py --mode infer --backend hf \
    --metadata meta.csv --config feature_prompts.yaml --out results.csv
```

## Input

Metadata CSV with columns:
`case_id, feature_name, image_paths (';'-separated, 1-4), modality, plane, ground_truth_label`
(this is exactly what the preprocessing pipeline's `metadata.csv` produces).

Feature vocab + prompt wording live in `feature_prompts.yaml` — not hardcoded.
`build_prompt` is only a **structural assembler**: it supplies the general
imaging/clinical context and the strict answer format, and slots in each
feature's `description` → `label_definitions` (optional) → `task` from the YAML.
To tune wording for a feature, edit the YAML; to change the framing for *all*
features, edit the general block in `build_prompt`.

## Run

```bash
# 1. Sanity-check FIRST on a few known-easy cases (see model-card caveat):
python run_medgemma.py --mode infer --metadata easy_subset.csv \
    --config feature_prompts.yaml --out results_sanity.csv

# 2. Full run (resume-safe: re-running skips done case_id+feature_name pairs):
python run_medgemma.py --mode infer --metadata meta.csv \
    --config feature_prompts.yaml --out results.csv

# 3. Eval:
python run_medgemma.py --mode eval --results results.csv --config feature_prompts.yaml
```

`--model-id google/medgemma-27b-it` for a comparison run (verify a 1.5/multi-
slice 27B exists first — see `load_model`).

## Output

`results.csv`: `case_id, feature_name, num_images_used, per_image_labels,
raw_output, parsed_label, ground_truth_label, correct`.

Eval prints overall + per-feature accuracy, confusion matrices, and **accuracy by
`num_images_used`** — the diagnostic for whether aggregating more views helps or
hurts.

## Before trusting results

- **Sanity-check the model** on a handful of obvious cases — MedGemma's card
  notes multimodal eval is primarily single-image.
- **Verify the API** against the current model card: model class, chat-template
  message format, and JPEG recommendation are flagged inline in `run_medgemma.py`
  where I relied on Gemma-3 conventions that may shift between versions.
- MedGemma use is governed by the **HAI-DEF** license terms.
