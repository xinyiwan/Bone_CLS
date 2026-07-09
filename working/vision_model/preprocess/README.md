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
