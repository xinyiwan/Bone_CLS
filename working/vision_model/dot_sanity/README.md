# 3D ViT — synthetic-dot sanity experiment

A minimal, self-contained pipeline to **verify the 3D ViT implementation** (model +
data loader + preprocessing + augmentation + training loop) *before* the real
imaging features are collected.

## Idea

Inject a small bright sphere ("dot", intensity `0.95 * max` after preprocessing)
near the centre of half the volumes, label those `1` and the rest `0`, and train
the ViT to predict the label. A correctly wired pipeline reaches **val AUC ≈ 1.0**
within a few epochs. If it doesn't, the bug is in the plumbing — not the data —
which is exactly what we want to catch early.

## Files

| File | Role |
|------|------|
| `vit3d.py`        | Compact 3D ViT: `Conv3d` patch embed → `[CLS]` + transformer encoder → linear head. Optional `patch_padding_mask` for later variable-size tumour volumes. |
| `dataset.py`      | `SyntheticDotDataset` (no files needed) and `NiftiDotDataset` (real `.nii.gz`, GIST-style `[Z,Y,X]` + seg-crop). Dot injected **after** normalisation. |
| `inspect_data.py` | Sanity-checks the data *before* training: balance, separability-AUC of simple intensity features, and a max-projection montage. Run this first. |
| `train.py`        | Training loop + rank-based AUC + **PASS/FAIL** sanity verdict (exit code 0/1). |

## Why the signal must be the brightest, localized region

In `SyntheticDotDataset` the background is min-max normalised then **capped at
`--bg-ceiling` (default 0.6)**, and the dot is written at the absolute value
`--dot-frac` (default 0.95). Because `dot_frac > bg_ceiling`, the dot is the
unambiguous bright blob — a clean, learnable signal. (An earlier version set the
dot to `0.95 × max`, which was *dimmer* than the background's own peaks, so the
model couldn't learn anything — train loss stuck at ln 2 ≈ 0.69.)

Raise `--bg-ceiling` toward `--dot-frac` to make the task progressively harder.

## Run

```bash
pip install -r requirements.txt

# 0. Look at the data first (balance + separability + montage):
python inspect_data.py --n 64 --out inspect_out

# Primary sanity check — pure synthetic, runs anywhere:
python train.py --mode synthetic --epochs 15

# With augmentation on (rotations/flips/noise must not break learning):
python train.py --mode synthetic --epochs 20 --augment

# Real volumes with injected dots (CSV columns: img[,seg]):
python train.py --mode nifti --csv paths.csv --img-root /projects/.../niftis --epochs 15
```

Expected synthetic output ends with `SANITY CHECK: PASS ✅` and `best val AUC` near 1.0.

## Knobs worth turning (still part of the sanity check)

- `--dot-frac` lower (e.g. `0.6`) and `--radius 1` → harder signal; confirms the
  model isn't only solving the trivial case.
- `--augment` → confirms the augmentation pipeline preserves the label.
- `--img-size` / `--patch-size` → confirm patchify divisibility logic holds.

## How this becomes the real model

Once the radiologist-verified imaging signs are collected, the synthetic label is
swapped for a real one with **no change to `vit3d.py`**:
- binary sign (e.g. fluid-fluid level present) → keep `num_classes=1` + BCE;
- multi-sign → set `num_classes=K`, one BCE head per sign (multi-label);
- the `NiftiDotDataset` preprocessing path becomes the production loader (drop the
  `inject_dot` call, read the label from your annotation table).
