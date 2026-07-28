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

Usage (--seq-table is REQUIRED: sequence types come only from the reviewed
table, never from scan names; cases missing from it are skipped):
    python to_nnunet.py <root> --out $nnUNet_raw --dataset-id 501 \
        --dataset-name BoneTumour --seq-table clf_perf/combined_reviewed.csv
    # plus affine-derived plane labels (analyze_dataset.py's per_scan.csv):
    python to_nnunet.py <root> --out $nnUNet_raw --dataset-id 501 \
        --dataset-name BoneTumour --seq-table clf_perf/combined_reviewed.csv \
        --plane-table analysis_out/per_scan.csv

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
from nibabel.processing import resample_from_to

from pairs import (MissingSequence, find_pairs, load_plane_table,
                   load_sequence_table, resolve_plane, resolve_sequence)


def sanitize(s: str) -> str:
    """Make a path-safe token (keep alnum, collapse the rest to single '-')."""
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")


def session_date(s: str) -> str:
    """Extract the 8-digit YYYYMMDD date embedded in a session folder name."""
    m = re.search(r"(\d{8})", s)
    return m.group(1) if m else ""


def load_exclusions(path: Path) -> set:
    """Sessions to drop, from kira-0515-seg.csv (If_segmented == 'exclude').

    Excludes at the SESSION level — set of (subject_code, 'YYYYMMDD') — because
    some subjects have one 'done' and one 'exclude' session. The date comes from
    'fechaHoraRealizacion'; if absent we fall back to (subject, '') = whole
    subject. Rows with If_segmented blank/'done' are kept.

    Escalation to the SUBJECT level: if a subject's only row in the table is an
    'exclude' row, the whole subject is dropped — (subject, ''). The table does
    not list every session present on disk, so a lone 'exclude' row means the
    subject was rejected outright and its other sessions were never reviewed;
    session-level matching would silently let them through.
    """
    import csv
    excl, rows_per_subject = set(), {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            subj = str(r.get("subject_code", "")).strip()
            rows_per_subject[subj] = rows_per_subject.get(subj, 0) + 1
            if str(r.get("If_segmented", "")).strip().lower() != "exclude":
                continue
            m = re.search(r"(\d{4})-(\d{2})-(\d{2})", str(r.get("fechaHoraRealizacion", "")))
            date = "".join(m.groups()) if m else ""
            excl.add((subj, date))

    whole = {s for s, _ in excl if rows_per_subject.get(s, 0) == 1}
    if whole:
        print(f"  exclusions escalated to whole subject (single table row): "
              f"{len(whole)} -> {sorted(whole)}")
    excl |= {(s, "") for s in whole}
    return excl


def load_binarised_label(seg_path: Path, image_path: Path) -> nib.Nifti1Image:
    """Read a mask and return it binarised (>0 -> 1, uint8) ON THE IMAGE GRID.

    The masks are NOT stored on the image voxel grid, so we resample the mask
    into the image's space using both affines (nearest-neighbour, order=0) --
    this is the spatially-correct alignment. If the grids already match, this is
    a no-op. (Copying the image affine instead would silently misalign masks.)
    """
    img = nib.load(str(image_path))
    seg = nib.load(str(seg_path))
    seg_on_img = resample_from_to(seg, (img.shape[:3], img.affine), order=0)
    binary = (np.asanyarray(seg_on_img.dataobj) > 0).astype(np.uint8)
    return nib.Nifti1Image(binary, img.affine)


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
    ap.add_argument("--seq-table", type=Path, required=True,
                    help="clf_perf/combined_reviewed.csv -- the only source of "
                         "sequence types (required; no filename fallback)")
    ap.add_argument("--plane-table", type=Path, default=None,
                    help="analyze_dataset.py per_scan.csv (for affine-derived plane)")
    ap.add_argument("--exclude-table", type=Path, default=None,
                    help="CSV of subjects/scans to drop (e.g. lesion < 10mm)")
    args = ap.parse_args()

    seq_lookup = load_sequence_table(args.seq_table)
    print(f"loaded sequence table: {len(seq_lookup)} entries")

    plane_lookup = None
    if args.plane_table:
        plane_lookup = load_plane_table(args.plane_table)
        print(f"loaded plane table: {len(plane_lookup)} entries")

    exclude = load_exclusions(args.exclude_table) if args.exclude_table else set()
    if exclude:
        print(f"loaded exclusions: {len(exclude)} subjects (e.g. lesion < 10mm)")

    ds_dir = args.out / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    images_dir = ds_dir / "imagesTr"
    labels_dir = ds_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    meta = []                       # per-case metadata rows
    seen = {}                       # case_id -> seg_path (collision guard)
    n_seg = n_missing = n_empty = n_excluded = n_no_seq = 0

    for subject, session, scan, image_path, seg_path, seg_source in find_pairs(args.root):
        n_seg += 1
        if (subject, session_date(session)) in exclude or (subject, "") in exclude:
            n_excluded += 1
            continue
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

        # Sequence type is mandatory and comes only from the reviewed table.
        try:
            seq = resolve_sequence(scan, subject, seq_lookup)
        except MissingSequence as e:
            n_no_seq += 1
            print(f"  [WARN] {e}; skipping {seg_path}")
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

        img_hdr = nib.load(str(image_path))
        plane = resolve_plane(scan, subject, plane_lookup,
                              img_hdr.affine, img_hdr.header.get_zooms()[:3])
        meta.append(dict(case=case_id, subject=subject, session=session,
                         scan=scan, sequence=seq, plane=plane,
                         seg_source=seg_source, fg_voxels=n_fg))
        seen[case_id] = seg_path

    if not meta:
        raise SystemExit("no usable cases built — check paths / layout")

    n = len(meta)
    from collections import Counter
    print(f"\nfound {n_seg} masks; {n_excluded} excluded; {n_missing} no image; "
          f"{n_no_seq} no sequence-table entry; {n_empty} empty; "
          f"{n} cases written to {ds_dir}")
    print("mask source:", dict(Counter(m["seg_source"] for m in meta)))

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
    print("\nsequence:", dict(Counter(m["sequence"] for m in meta)))
    print("plane:   ", dict(Counter(m["plane"] for m in meta)))
    print(f"\nNext:\n  nnUNetv2_plan_and_preprocess -d {args.dataset_id} "
          f"--verify_dataset_integrity\n"
          f"  cp {ds_dir/'splits_final.json'} "
          f"$nnUNet_preprocessed/{ds_dir.name}/splits_final.json\n"
          f"  nnUNetv2_train {args.dataset_id} 3d_fullres 0")


if __name__ == "__main__":
    main()
