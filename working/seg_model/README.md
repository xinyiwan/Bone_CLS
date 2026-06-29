# Bone-tumour segmentation

Train a segmentation model on the ~200 manually-segmented cases (nnU-Net baseline).

## Data layout

```
<root>/<subject>/<session>/<scan>/images.nii.gz
                          /segmentation_history/segs/<scan>_seg.nii.gz   <- mask used
                          /segmentation_history/FINAL_* | point_* | lasso_*  <- ignored (history)
```

Masks come from `segmentation_history/segs/`. In a typical session the lesion is
segmented on **every sequence**, so each `(sequence image, its seg)` is a labelled
pair. Discovery is **segmentation-driven**: the `segs/` masks define the labelled
set (some images are excluded during segmentation) and each mask is traced back to
its `images.nii.gz`; unsegmented images are never visited.

## Steps

1. **`analyze_dataset.py` — distribution analysis (do this first).**
   Per labelled scan: plane (from affine), sequence (table or filename), per-axis
   size/spacing, intensity stats (whole-image + in-lesion), tumour volume, label
   content. Writes `per_scan.csv`, `summary.txt`, and `plots/`.

   ```bash
   python analyze_dataset.py <root> --out-dir analysis_out
   # with reviewed sequence types (composite of Clase W/FS/C Final):
   python analyze_dataset.py <root> --out-dir analysis_out \
       --seq-table ../../clf_perf/combined_reviewed.csv
   ```

2. **`to_nnunet.py` — convert to nnU-Net v2 raw format.**
   Seg-driven; binarises masks (>0→1); checks image/mask geometry; writes
   `imagesTr/<case>_0000.nii.gz`, `labelsTr/<case>.nii.gz`, `dataset.json`,
   `case_metadata.csv` (case→subject/sequence/plane), and a **subject-level
   GroupKFold** `splits_final.json`.

   ```bash
   export nnUNet_raw=... nnUNet_preprocessed=... nnUNet_results=...
   python to_nnunet.py <root> --out $nnUNet_raw --dataset-id 501 \
       --dataset-name BoneTumour \
       --seq-table ../../clf_perf/combined_reviewed.csv   # true sequence types
   ```

3. **Train nnU-Net** — fingerprint + preprocess (auto normalisation/spacing/patch),
   then train each fold. **Copy our subject-level split over nnU-Net's random one.**

   ```bash
   nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity
   cp $nnUNet_raw/Dataset501_BoneTumour/splits_final.json \
      $nnUNet_preprocessed/Dataset501_BoneTumour/splits_final.json   # subject-level!
   for f in 0 1 2 3 4; do nnUNetv2_train 501 3d_fullres $f; done
   ```

4. **Evaluate per sequence/plane** — join nnU-Net's per-case Dice with
   `case_metadata.csv` to report Dice broken down by sequence and plane (the key
   result for whether pooled sequence-agnostic training works here).

## Design decisions baked in

- **binary lesion**, single channel (`MR`), **pooled sequence-agnostic** (each
  independently-corrected per-sequence mask is one case).
- **subject-level GroupKFold** — `to_nnunet.py` emits `splits_final.json`; you
  must copy it into `nnUNet_preprocessed/<dataset>/` so subjects don't leak.
- masks are independently corrected per sequence (not just registration-
  propagated), so pooling adds real label information, not duplicates.

## Why analysis comes first

nnU-Net auto-configures normalisation/spacing/patch size, but it **cannot** decide:
- **single-sequence vs pooled vs multi-channel** training — sequences here have
  *different planes/grids* (SAG vs COR vs AX), so they cannot be stacked as
  co-registered channels; they must be pooled as independent training cases or a
  per-sequence model trained. The sequence × plane crosstab drives this.
- **subject-level splitting** — multiple sequences of one subject must not leak
  across train/val folds.
