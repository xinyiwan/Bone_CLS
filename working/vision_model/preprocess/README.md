# 2D crop extraction for MedGemma (128x128)

Turn 3D MRI volumes + nnInteractive masks into 128x128 8-bit PNG crops for a
vision-language model, driven by a feature -> modality -> plane -> margin config.

There is **one entry point, `run.py`**. It reads the segmentation project's real
on-disk layout via `seg_model/pairs.py:find_pairs` and takes the sequence label
per scan from your classified-sequence table.

## Install (no root needed)

```bash
pip install --user nibabel numpy opencv-python pandas matplotlib pyyaml
```

## Data layout (as produced by the segmentation project)

```
<root>/<subject>/<session>/<scan>/images.nii.gz                        # volume
<root>/<subject>/<session>/segmentation_history/segs/<scan>_seg.nii.gz # mask (history)
<root>/<subject>/<session>/review/<xxx>/segs/<scan>_seg.nii.gz         # mask (reviewed, preferred)
```

Discovery is **segmentation-driven**: only scans that have a mask are considered.
The **sequence** (T1W_FS_CE, T2W_FS, ...) comes from your classified-sequence
table joined on `(subject, session, scan)`; the **acquisition plane** is read
from each scan's affine. A `(sequence, plane)` requirement is matched to a scan
**acquired in that plane** and sliced along its native axis — we never reslice a
thick axial stack into a fake coronal.

## Step 1 — see what's available and how the config is covered (no extraction)

```bash
# This project's sequence table names the subject column 'case':
python run.py --data-root /data --out-root ./out \
    --sequence-table sequences.csv --config feature_config.yaml \
    --seq-subject-col case --index-only
```

Writes `out/dataset_index.csv` (provenance) and prints the available
`(sequence, plane)` combos + per-feature coverage. Use the exact label strings
it prints in `feature_config.yaml` (directly, or via `sequence_aliases:`).

## Step 2 — extract one subject first, with overlays, then eyeball it

```bash
python run.py --data-root /data --out-root ./out \
    --sequence-table sequences.csv --config feature_config.yaml \
    --seq-subject-col case --subjects SUBJ001 --overlay

python qc_contact_sheet.py ./out/metadata.csv --n 12 --out contact_sheet.png
```

## Step 3 — full batch (with ground-truth labels)

```bash
python run.py --data-root /data --out-root ./out \
    --sequence-table sequences.csv --config feature_config.yaml --seq-subject-col case \
    --labels-dir ../label/label_out/jsons
```

Per-case / per-feature errors and missing `(sequence, plane)` combinations are
logged and skipped; the batch continues.

Outputs:
```
out/{case_id}/{feature}/{modality}_{plane}_{slice}.png
out/{case_id}/{feature}/{modality}_{plane}_{slice}_overlay.png   # if --overlay
out/metadata.csv          # one row per image/plane (flattened: no mixing of orientations)
out/dataset_index.csv     # provenance: which scan/source each crop came from
```

**Metadata columns:**
`case_id, feature_name, modality, plane, slice_index, image_path, crop_bbox, margin_used, ground_truth_label`

Each row is a single image. This **flattened structure** keeps orientations (axial/coronal/sagittal) separate for cleaner downstream processing — the VLM sees one image at a time, and you can inspect per-image results before aggregating.

## Ground-truth labels (`--labels-dir`)

Ground truth lives in per-subject **assessment JSONs** — the ones
`label/json_extract.py` selects and copies to `<out-dir>/jsons/<subject>.json`.
Point `run.py` at that folder with `--labels-dir`, and each row's
`ground_truth_label` is filled from the subject's assessment:

```
<subject>.json  ->  imaging_features[<assessment_key>]  ->  ground_truth_label
```

- The **feature → assessment key** mapping is `assessment_key:` in the feature
  config (e.g. `shape` → `tumor_shape`). Omit it when the names already match.
- A **missing** file, features block, or key → `ground_truth_label = "unknown"`
  (not every subject is labelled yet). Downstream eval skips `unknown` rows.
- **List** values (e.g. `tumor_matrix_mri: ["Cartilaginous"]`) are `;`-joined.
- Values are copied **verbatim** (e.g. `Round/Oval`, `Absence / unknown`); if you
  need them normalized to your prompt `label_options` for scoring, do that on the
  eval side.

Omit `--labels-dir` entirely and every row is labelled `unknown` (still a valid
column, just unscored). Change the top-level key with `--imaging-features-key`
if your JSON nests features under a different name.

## Key options

- `--seq-*-col` map to your table's column names (defaults `subject` / `session`
  / `scan` / `sequence`). This project uses `case` for the subject, so pass
  `--seq-subject-col case`.
- `--unit subject` (default; `case_id` == subject) vs `--unit study`
  (`case_id` == subject/session, so a feature's axial + coronal are guaranteed
  from the same study).
- `--crop-mode bbox` (default, keeps surrounding tissue) vs `masked` (zero
  everything outside the segmentation); `--mask-dilate-px N` keeps an N-px rim in
  masked mode. Also settable per-feature in the config (`crop_mode:`).
- `--reviewed-only` restricts to radiologist-reviewed masks; otherwise reviewed
  is preferred and history is the fallback.
- `--labels-dir DIR` fills `ground_truth_label` from assessment JSONs at
  `DIR/<subject>.json` (the `label/json_extract.py` output). Missing → `unknown`.
  `--imaging-features-key` (default `imaging_features`) sets the JSON block read.
- `--norm minmax|zscore`, `--pad-mode clip|pad`, `--no-foreground-norm`,
  `--out-size` (default 128).

## Pipeline steps (each is a standalone, testable function)

| Step        | Module         | Key function |
|-------------|----------------|--------------|
| discover + match (subject/session/scan → sequence, plane) | `run.py` | `build_index`, `make_dataset_resolver` |
| load + orient (RAS) | `volume_io.py` | `load_volume_and_mask` |
| slice selection     | `slicing.py`   | `find_max_area_slice`, `find_top_k_area_slices` |
| normalization       | `normalize.py` | `normalize_intensity` (swap this out freely) |
| cropping (px / mm)  | `cropping.py`  | `mask_bbox_2d`, `expand_bbox`, `crop_with_bbox`, `apply_mask` |
| resize              | `resize.py`    | `resize_image` (bilinear), `resize_mask` (nearest) |
| overlay QC          | `overlay.py`   | `draw_contour_overlay` |
| glue                | `pipeline.py`  | `process_feature`, `process_case` (resolver-injected) |

Quick unit test example (no data needed):
```python
import numpy as np
from slicing import find_max_area_slice
m = np.zeros((10, 10, 5), int); m[2:8, 2:8, 3] = 1   # biggest blob on axial slice 3
assert find_max_area_slice(m, "axial") == 3
```

## Things to double-check on YOUR data (see inline flags)

- **Scan-name join:** the `scan` value in the sequence table must equal the
  scan-folder name `find_pairs` yields. After `--index-only`, check the log line
  `"N had no label"` — if high, the join is off and few crops will be produced.
- **Orientation:** everything is reoriented to canonical RAS. For *oblique*
  acquisitions the plane is only approximately axial/coronal/sagittal — verify
  with `--overlay` across a few scanners before trusting the batch.
- **Mask alignment:** masks are assumed to share the volume's grid/affine. An
  affine-mismatch-but-same-shape case is *warned*, not fixed — add resampling
  (`nibabel.processing.resample_from_to`) if you see those warnings.
- **Background normalization:** percentiles use non-zero voxels as a foreground
  proxy (`--no-foreground-norm` to disable). If your background isn't ~0, revisit.
- **`--unit subject` + multiple studies:** a feature's axial and coronal could
  then come from different studies; use `--unit study` if that matters.
