"""
Convert the bone-tumour tree into an nnU-Net v2 raw dataset.

Design (decided with the data owner):
  * binary lesion segmentation (label 1 = tumour; masks binarised >0 -> 1)
  * pooled, sequence-agnostic: each independently-corrected per-sequence mask is
    one training case (single channel, modality "MR")
  * subject-level GroupKFold splits (no subject leaks across folds), written as
    splits_final.json for nnU-Net to consume
  * case ids encode subject + scan so per-sequence / per-plane Dice can be
    recovered later (see case_metadata.csv)

Output (nnU-Net v2 raw layout):
    <out>/Dataset<ID>_<NAME>/
        imagesTr/<case>_0000.nii.gz      (copy of images.nii.gz)
        labelsTr/<case>.nii.gz           (binarised mask, uint8, image geometry)
        dataset.json
        splits_final.json                (subject-level K-fold; copy to preprocessed)
        case_metadata.csv                (case -> subject, sequence, plane, ...)

Usage:
    python to_nnunet.py <root> --out $nnUNet_raw --dataset-id 501 \
        --dataset-name BoneTumour
    # with reviewed sequence types (clf_perf/combined_reviewed.csv):
    python to_nnunet.py <root> --out $nnUNet_raw --dataset-id 501 \
        --dataset-name BoneTumour --seq-table clf_perf/combined_reviewed.csv

Then:
    nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity
    cp <out>/Dataset501_BoneTumour/splits_final.json \
       $nnUNet_preprocessed/Dataset501_BoneTumour/splits_final.json
    nnUNetv2_train 501 3d_fullres 0   # (folds 0..4)
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import nibabel as nib

from pairs import find_pairs, load_sequence_table, plane_from_name, resolve_sequence


def sanitize(s: str) -> str:
    """Make a path-safe token (keep alnum, collapse the rest to single '-')."""
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")


def load_binarised_label(seg_path: Path, image_path: Path) -> nib.Nifti1Image:
    """Read a mask, binarise (>0 -> 1, uint8), put it on the image's grid.

    The mask is corrected on the scan's own grid, so it has the same shape as
    the image; writing it with the image's affine guarantees nnU-Net sees
    identical geometry (no sub-voxel header drift). Raises if shapes differ.
    """
    img = nib.load(str(image_path))
    seg_arr = np.asanyarray(nib.load(str(seg_path)).dataobj)
    if seg_arr.shape != img.shape[:3]:
        raise ValueError(f"shape mismatch: image {img.shape[:3]} vs "
                         f"mask {seg_arr.shape} for {seg_path.name}")
    binary = (seg_arr > 0).astype(np.uint8)
    return nib.Nifti1Image(binary, img.affine)        # clean uint8 header from affine


def subject_group_kfold(subjects, k: int, seed: int):
    """Assign whole subjects to folds (round-robin after shuffle).

    Guarantees no subject appears in two folds. Returns {fold: set(subjects)}.
    """
    subs = sorted(set(subjects))
    rng = np.random.default_rng(seed)
    rng.shuffle(subs)
    folds = {f: set() for f in range(k)}
    for i, s in enumerate(subs):
        folds[i % k].add(s)
    return folds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="nnUNet_raw root")
    ap.add_argument("--dataset-id", type=int, required=True)
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seq-table", type=Path, default=None,
                    help="clf_perf/combined_reviewed.csv (for true sequence type)")
    args = ap.parse_args()

    seq_lookup = None
    if args.seq_table:
        seq_lookup = load_sequence_table(args.seq_table)
        print(f"loaded sequence table: {len(seq_lookup)} entries")

    ds_dir = args.out / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    images_dir = ds_dir / "imagesTr"
    labels_dir = ds_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    meta = []                       # per-case metadata rows
    seen = {}                       # case_id -> seg_path (collision guard)
    n_seg = n_missing = n_empty = 0

    for subject, session, scan, image_path, seg_path in find_pairs(args.root):
        n_seg += 1
        if image_path is None:
            n_missing += 1
            print(f"  [WARN] no image for mask {seg_path}")
            continue

        case_id = f"{sanitize(subject)}__{sanitize(scan)}"
        if case_id in seen:
            # Same subject+scan from two sessions -> disambiguate with session.
            case_id = f"{case_id}__{sanitize(session)}"
        if case_id in seen:
            print(f"  [WARN] duplicate case id {case_id}; skipping {seg_path}")
            continue

        try:
            label = load_binarised_label(seg_path, image_path)
        except ValueError as e:
            print(f"  [WARN] {e}")
            continue

        n_fg = int(np.asanyarray(label.dataobj).sum())
        if n_fg == 0:
            n_empty += 1
            print(f"  [WARN] empty mask after binarisation: {seg_path}; skipping")
            continue

        # copy image (preserves geometry exactly), write binarised label
        shutil.copyfile(image_path, images_dir / f"{case_id}_0000.nii.gz")
        nib.save(label, str(labels_dir / f"{case_id}.nii.gz"))

        seq = resolve_sequence(scan, subject, seq_lookup)
        meta.append(dict(case=case_id, subject=subject, session=session,
                         scan=scan, sequence=seq, plane=plane_from_name(scan),
                         fg_voxels=n_fg))
        seen[case_id] = seg_path

    if not meta:
        raise SystemExit("no usable cases built — check paths / layout")

    n = len(meta)
    print(f"\nfound {n_seg} masks; {n_missing} no image; {n_empty} empty; "
          f"{n} cases written to {ds_dir}")

    # dataset.json (nnU-Net v2 schema)
    dataset_json = {
        "channel_names": {"0": "MR"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": n,
        "file_ending": ".nii.gz",
        "name": args.dataset_name,
        "description": "Bone tumour lesion segmentation, pooled sequence-agnostic.",
    }
    (ds_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2))

    # case_metadata.csv
    import csv
    with open(ds_dir / "case_metadata.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(meta[0].keys()))
        w.writeheader()
        w.writerows(meta)

    # subject-level K-fold -> splits_final.json
    cases_by_subject = {}
    for m in meta:
        cases_by_subject.setdefault(m["subject"], []).append(m["case"])
    folds = subject_group_kfold([m["subject"] for m in meta], args.folds, args.seed)
    splits = []
    for f in range(args.folds):
        val_subjects = folds[f]
        val = [c for s in val_subjects for c in cases_by_subject[s]]
        train = [m["case"] for m in meta if m["subject"] not in val_subjects]
        splits.append({"train": sorted(train), "val": sorted(val)})
        print(f"  fold {f}: {len(val_subjects)} subjects / {len(val)} val cases, "
              f"{len(train)} train cases")
    (ds_dir / "splits_final.json").write_text(json.dumps(splits, indent=2))

    # sequence / plane distribution (sanity)
    from collections import Counter
    print("\nsequence:", dict(Counter(m["sequence"] for m in meta)))
    print("plane:   ", dict(Counter(m["plane"] for m in meta)))
    print(f"\nNext:\n  nnUNetv2_plan_and_preprocess -d {args.dataset_id} "
          f"--verify_dataset_integrity\n"
          f"  cp {ds_dir/'splits_final.json'} "
          f"$nnUNet_preprocessed/{ds_dir.name}/splits_final.json\n"
          f"  nnUNetv2_train {args.dataset_id} 3d_fullres 0")


if __name__ == "__main__":
    main()
