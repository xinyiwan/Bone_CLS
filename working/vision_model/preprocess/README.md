# 2D crop extraction for MedGemma (128x128)

Turn 3D MRI volumes + nnInteractive masks into 128x128 8-bit PNG crops for a
vision-language model, driven by a feature -> modality -> plane -> margin table.

## Install (no root needed)

```bash
pip install --user nibabel numpy opencv-python pandas matplotlib pyyaml
```

## Data layout (default; change with --img-pattern / --seg-pattern)

```
/data/{subject_id}/{modality}.nii.gz          # volume
/data/{subject_id}/{modality}_seg.nii.gz      # mask, same grid as its volume
```

## Run on ONE subject first, with overlays, then eyeball it

```bash
python run.py --data-root /data --out-root ./out \
    --config feature_config.yaml --subjects SUBJ001 --overlay

python qc_contact_sheet.py ./out/metadata.csv --n 12 --out contact_sheet.png
```

Outputs:
```
out/{subject}/{feature}/{modality}_{plane}_{slice}.png
out/{subject}/{feature}/{modality}_{plane}_{slice}_overlay.png   # if --overlay
out/metadata.csv
```

## Full batch

```bash
python run.py --data-root /data --out-root ./out \
    --config feature_config.yaml --subject-list subjects.txt
```

Missing modalities/masks and per-feature errors are logged and skipped; the
batch continues.

## Running on the REAL BONE data layout (recommended)

`run.py` assumes a flat `{modality}.nii.gz` layout. For the actual segmentation
project tree — `<subject>/<session>/<scan>/images.nii.gz` with masks under
`segmentation_history/segs/` or `review/*/segs/` — use **`extract_from_dataset.py`**,
which reuses `seg_model/pairs.py:find_pairs` (segmentation-driven discovery,
reviewed mask preferred over history) and joins your **classified-sequence
table** on `(subject, session, scan)` for the sequence label. The acquisition
plane is read from each scan's affine; a `(sequence, plane)` requirement is
matched to a scan **acquired in that plane** and sliced along its native axis —
we never reslice a thick axial stack into a fake coronal.

The unit of work is the **subject** by default (`case_id` == subject); pass
`--unit study` to group by `(subject, session)` instead so a feature's
`axial + coronal` are guaranteed to come from the same study.

```bash
# 1. Discover what's available and how well the config is covered (no extraction):
#    NOTE: this project's table names the subject column 'case' -> --seq-subject-col case
python extract_from_dataset.py --data-root /data --out-root ./out \
    --sequence-table sequences.csv --config feature_config_dataset.yaml --index-only \
    --seq-subject-col case --seq-session-col session \
    --seq-scan-col scan --seq-label-col sequence

# -> writes out/dataset_index.csv (provenance) and prints available
#    (sequence, plane) combos + per-feature coverage. Use the exact label
#    strings it prints in feature_config_dataset.yaml (or via sequence_aliases).

# 2. Extract:
python extract_from_dataset.py --data-root /data --out-root ./out \
    --sequence-table sequences.csv --config feature_config_dataset.yaml \
    --seq-subject-col case --overlay --reviewed-only
```

Notes:
- `--seq-*-col` flags map to your table's actual column names (defaults:
  `subject`, `session`, `scan`, `sequence`). Your table uses `case` for the
  subject, so pass `--seq-subject-col case`.
- `--unit subject` (default) vs `--unit study` picks the grouping granularity.
- `sequence_aliases:` in the YAML lets the feature config use short names
  (`T1C`) that map to your classifier's labels (`T1W_nFS_CE`).
- `--reviewed-only` restricts to radiologist-reviewed masks; otherwise reviewed
  is preferred and history used as fallback.
- Provenance (which scan/source each crop came from) is in `dataset_index.csv`,
  keyed by the same image paths that appear in `metadata.csv`.

## Pipeline steps (each is a standalone, testable function)

| Step        | Module         | Key function |
|-------------|----------------|--------------|
| load + orient (RAS) | `volume_io.py` | `load_volume_and_mask` |
| slice selection     | `slicing.py`   | `find_max_area_slice`, `find_top_k_area_slices` |
| normalization       | `normalize.py` | `normalize_intensity` (swap this out freely) |
| cropping (px / mm)  | `cropping.py`  | `mask_bbox_2d`, `expand_bbox`, `crop_with_bbox` |
| resize              | `resize.py`    | `resize_image` (bilinear), `resize_mask` (nearest) |
| overlay QC          | `overlay.py`   | `draw_contour_overlay` |

Quick unit test example:
```python
import numpy as np
from slicing import find_max_area_slice
m = np.zeros((10, 10, 5), int); m[2:8, 2:8, 3] = 1   # biggest blob on axial slice 3
assert find_max_area_slice(m, "axial") == 3
```

## Things to double-check on YOUR data (see inline flags)

- **Orientation:** everything is reoriented to canonical RAS. For *oblique*
  acquisitions the resulting plane is only approximately axial/coronal/sagittal
  — verify with `--overlay` across a few scanners before trusting the batch.
- **Mask alignment:** masks are assumed to share the volume's grid/affine. An
  affine-mismatch-but-same-shape case is *warned*, not fixed — add resampling
  (`nibabel.processing.resample_from_to`) if you see those warnings.
- **Background normalization:** percentiles use non-zero voxels as a foreground
  proxy (`--no-foreground-norm` to disable). If your background isn't ~0, revisit.
