"""
Extract 2D VLM crops (128x128 PNGs) from the segmentation project's on-disk
layout -- the same tree walked by seg_model/pairs.py and analysis/seg_history_overview.py:

    <root>/<subject>/<session>/<scan>/images.nii.gz
    <root>/<subject>/<session>/segmentation_history/segs/<scan>_seg.nii.gz   (history)
    <root>/<subject>/<session>/review/<xxx>/segs/<scan>_seg.nii.gz           (reviewed)

This is the single entry point. Discovery is segmentation-driven via
pairs.find_pairs, so only scans that have a mask are considered (a reviewed mask
wins over a history one).

Sequence type comes from YOUR classified-sequence table (one final label per
scan), joined on (subject, session, scan). The acquisition PLANE is read from
each scan's affine (pairs.plane_from_affine).

The feature config asks for (modality/sequence, plane). We match it to a scan
ACQUIRED in that plane and slice along its native slice axis -- we never reslice
a thick axial stack into a fake coronal. Missing (sequence, plane) combinations
for a case are logged and skipped.

Unit of work is the SUBJECT by default (--unit subject): one metadata row group
per patient, so `case_id` in the output == subject. If a subject has more than
one study, a feature's axial and coronal may then come from different studies;
pass `--unit study` to key on (subject, session) instead and keep every feature
within one study. The sequence table is always joined on (subject, session, scan)
regardless of --unit.

Usage:
  # 1. Discover what's available so you can author the feature config
  #    (this project's table names the subject column 'case'):
  python run.py --data-root /data --out-root ./out \\
      --sequence-table sequences.csv --config feature_config.yaml \\
      --seq-subject-col case --index-only

  # 2. Extract (start with one subject, add --overlay to QC):
  python run.py --data-root /data --out-root ./out \\
      --sequence-table sequences.csv --config feature_config.yaml \\
      --seq-subject-col case --subjects SUBJ001 --overlay

  # 3. Full batch, then contact sheet:
  python run.py --data-root /data --out-root ./out \\
      --sequence-table sequences.csv --config feature_config.yaml --seq-subject-col case
  python qc_contact_sheet.py ./out/metadata.csv --n 24 --out contact_sheet.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from config import load_feature_config, load_sequence_aliases
from outputs import MetadataWriter
from pipeline import PipelineOptions, process_case

# Reuse the segmentation project's discovery + plane inference (same as
# seg_history_overview.py does), so this stays consistent with the rest.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "working" / "seg_model"))

log = logging.getLogger("preprocess")

# A discovered scan with everything we need to match + process it.
Record = Dict[str, str]


def make_case_id(subject: str, session: str, unit: str = "subject") -> str:
    """Identifier for the output `case_id`. 'subject' -> one group per patient;
    'study' -> per (subject, session)."""
    if unit == "subject" or not session:
        return subject
    return f"{subject}/{session}"


def load_sequence_labels(
    path: Path, subj_col: str, sess_col: str, scan_col: str, label_col: str
) -> Dict[Tuple[str, str, str], str]:
    """Build {(subject, session, scan): final_label} from the classified table."""
    df = pd.read_csv(path, dtype=str, low_memory=False).fillna("")
    for c in (subj_col, sess_col, scan_col, label_col):
        if c not in df.columns:
            raise SystemExit(f"{path}: missing column {c!r} (have: {list(df.columns)})")
    out: Dict[Tuple[str, str, str], str] = {}
    for _, r in df.iterrows():
        key = (r[subj_col].strip(), r[sess_col].strip(), r[scan_col].strip())
        label = r[label_col].strip()
        if label:
            out[key] = label
    return out


def build_index(
    data_root: Path,
    seq_labels: Dict[Tuple[str, str, str], str],
    reviewed_only: bool,
    unit: str = "subject",
) -> Tuple[Dict[str, Dict[Tuple[str, str], Record]], List[Record]]:
    """Walk the tree and index scans by case -> {(SEQUENCE_UPPER, plane): record},
    preferring reviewed masks. Also returns a flat provenance list."""
    from pairs import find_pairs, plane_from_affine  # lazy: needs nibabel
    import nibabel as nib

    index: Dict[str, Dict[Tuple[str, str], Record]] = defaultdict(dict)
    rows: List[Record] = []
    n_no_label = n_no_image = 0

    for subject, session, scan, image_path, seg_path, source in find_pairs(data_root):
        if image_path is None:
            n_no_image += 1
            continue
        if reviewed_only and source != "reviewed":
            continue

        label = seq_labels.get((subject, session, scan))
        if not label:
            n_no_label += 1
            log.debug("no sequence label for %s/%s/%s -- skipping", subject, session, scan)
            continue

        try:
            img = nib.load(str(image_path))
            plane = plane_from_affine(img.affine, img.header.get_zooms())
        except Exception as e:  # noqa: BLE001
            log.warning("header read failed for %s/%s: %s", subject, scan, e)
            continue

        case = make_case_id(subject, session, unit)
        rec: Record = {
            "case_id": case, "subject": subject, "session": session, "scan": scan,
            "sequence": label, "plane": plane, "source": source,
            "image_path": str(image_path), "seg_path": str(seg_path),
        }
        rows.append(rec)

        key = (label.upper(), plane)
        cur = index[case].get(key)
        # Prefer reviewed; break ties deterministically by scan name.
        if (cur is None
                or (cur["source"] != "reviewed" and source == "reviewed")
                or (cur["source"] == source and scan < cur["scan"])):
            index[case][key] = rec

    log.info("indexed %d scan(s) across %d case(s); %d had no label, %d had no image",
             len(rows), len(index), n_no_label, n_no_image)
    return index, rows


def make_dataset_resolver(index, aliases: dict):
    """(case_id, modality, plane) -> (image_path, seg_path) using the index."""

    def resolve(case_id: str, modality: str, plane: str):
        want = str(aliases.get(modality, modality)).upper()
        rec = index.get(case_id, {}).get((want, plane))
        return (Path(rec["image_path"]), Path(rec["seg_path"])) if rec else None

    return resolve


def write_index_csv(rows: List[Record], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["case_id", "subject", "session", "scan", "sequence", "plane", "source", "image_path", "seg_path"]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
    log.info("wrote provenance index -> %s", path)


def merge_clinical(meta_path: Path, clinical_csv: Path, key_col: str, cols: List[str]) -> None:
    """Post-step: left-join per-subject clinical fields into metadata.csv so the
    downstream prompt builder has anatomical location etc. without a second file.
    Joins on subject = case_id before any '/' (so it works for --unit study too)."""
    meta = pd.read_csv(meta_path, dtype=str).fillna("")
    clin = pd.read_csv(clinical_csv, dtype=str).fillna("")
    if key_col not in clin.columns:
        raise SystemExit(f"{clinical_csv}: no key column {key_col!r} (have {list(clin.columns)})")
    keep = [c for c in cols if c in clin.columns]
    missing = [c for c in cols if c not in clin.columns]
    if missing:
        log.warning("clinical CSV missing requested columns %s (have %s)", missing, list(clin.columns))
    meta["_subject"] = meta["case_id"].str.split("/").str[0]
    clin = clin.drop_duplicates(subset=key_col)[[key_col] + keep]
    out = meta.merge(clin, left_on="_subject", right_on=key_col, how="left").drop(columns=["_subject", key_col])
    out.to_csv(meta_path, index=False)
    log.info("merged clinical columns %s into %s", keep, meta_path)


def make_gt_lookup(labels_dir: Path, features_key: str = "imaging_features"):
    """Build a (case_id, FeatureSpec) -> ground-truth-label lookup.

    Reads per-subject assessment JSONs at ``<labels_dir>/<subject>.json`` (the
    output of label/json_extract.py) and pulls the label for a feature from the
    ``features_key`` block, using ``spec.assessment_key`` (falling back to
    ``spec.name``) as the key -- e.g. feature ``shape`` -> ``tumor_shape``.
    Returns "unknown" when the file, the features block, or the key is
    missing/empty. List values (e.g. tumor_matrix_mri) are ';'-joined.
    """
    import json

    cache: Dict[str, dict] = {}

    def features_for(subject: str) -> dict:
        if subject not in cache:
            path = labels_dir / f"{subject}.json"
            feats: dict = {}
            if path.exists():
                try:
                    with open(path, encoding="utf-8") as fh:
                        feats = json.load(fh).get(features_key, {}) or {}
                except Exception as e:  # noqa: BLE001
                    log.warning("could not read assessment %s: %s", path, e)
            else:
                log.debug("no assessment JSON for subject %s at %s", subject, path)
            cache[subject] = feats
        return cache[subject]

    def gt_for(case_id: str, spec) -> str:
        feats = features_for(case_id.split("/")[0])  # subject = case_id before any '/'
        val = feats.get(spec.assessment_key or spec.name)
        if val is None or val == "" or val == []:
            return "unknown"
        if isinstance(val, list):
            return ";".join(str(v) for v in val)
        return str(val)

    return gt_for


def print_availability(rows: List[Record]) -> None:
    combos = Counter((r["sequence"], r["plane"]) for r in rows)
    print("\nAvailable (sequence, plane) across all cases:")
    for (seq, plane), n in sorted(combos.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {n:5d}  {seq:<16} {plane}")


def print_coverage(index, specs, aliases) -> None:
    """For each feature, how many cases satisfy ALL its (modality, plane) reqs."""
    resolve = make_dataset_resolver(index, aliases)
    print("\nFeature coverage (cases fully satisfying every requirement):")
    for spec in specs:
        need = [(req.modality, p) for req in spec.requirements for p in req.planes]
        full = part = 0
        for case in index:
            got = sum(resolve(case, m, p) is not None for m, p in need)
            if got == len(need):
                full += 1
            elif got > 0:
                part += 1
        print(f"  {spec.name:<20} full={full:<5} partial={part:<5} (needs {need})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--config", type=Path, help="feature mapping (.yaml/.csv); required unless --index-only")
    ap.add_argument("--sequence-table", type=Path, required=True, help="classified-sequence CSV")
    ap.add_argument("--unit", choices=["subject", "study"], default="subject",
                    help="group output by patient (subject) or by study (subject+session)")
    # sequence-table column names (override to match your file; this project's
    # table names the subject column 'case' -> --seq-subject-col case)
    ap.add_argument("--seq-subject-col", default="subject")
    ap.add_argument("--seq-session-col", default="session")
    ap.add_argument("--seq-scan-col", default="scan")
    ap.add_argument("--seq-label-col", default="sequence")
    # case selection
    ap.add_argument("--subjects", nargs="*", help="restrict to these subject ids")
    ap.add_argument("--reviewed-only", action="store_true", help="use only radiologist-reviewed masks")
    ap.add_argument("--index-only", action="store_true", help="report availability/coverage; don't extract")
    ap.add_argument("--metadata", type=Path, help="metadata CSV (default out-root/metadata.csv)")
    # ground-truth labels from assessment JSONs (label/json_extract.py output)
    ap.add_argument("--labels-dir", type=Path,
                    help="folder of per-subject assessment JSONs (<subject>.json) for ground-truth "
                         "labels; missing labels -> 'unknown'. Feature->key mapping via assessment_key in the config.")
    ap.add_argument("--imaging-features-key", default="imaging_features",
                    help="top-level key in each assessment JSON holding the feature dict")
    # optional clinical-info merge (adds anatomical location etc. to metadata.csv)
    ap.add_argument("--clinical-csv", type=Path, help="per-subject clinical CSV (e.g. combine_cli_info.py output)")
    ap.add_argument("--clinical-key-col", default="subject", help="subject-id column in the clinical CSV")
    ap.add_argument("--clinical-cols", nargs="*",
                    default=["skeletal_location", "location_within_bone", "age", "gender"],
                    help="clinical columns to add to metadata.csv")
    # pipeline options
    ap.add_argument("--out-size", type=int, default=128)
    ap.add_argument("--norm", choices=["minmax", "zscore"], default="minmax")
    ap.add_argument("--pad-mode", choices=["clip", "pad"], default="clip")
    ap.add_argument("--crop-mode", choices=["bbox", "masked"], default="bbox")
    ap.add_argument("--mask-dilate-px", type=int, default=0)
    ap.add_argument("--no-foreground-norm", action="store_true")
    ap.add_argument("--overlay", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    seq_labels = load_sequence_labels(
        args.sequence_table, args.seq_subject_col, args.seq_session_col,
        args.seq_scan_col, args.seq_label_col,
    )
    log.info("loaded %d sequence label(s) from %s", len(seq_labels), args.sequence_table.name)

    index, rows = build_index(args.data_root, seq_labels, args.reviewed_only, args.unit)

    if args.subjects:
        keep = set(args.subjects)
        index = {c: v for c, v in index.items() if v and next(iter(v.values()))["subject"] in keep}
        rows = [r for r in rows if r["subject"] in keep]

    write_index_csv(rows, args.out_root / "dataset_index.csv")

    specs = load_feature_config(args.config) if args.config else []
    aliases = load_sequence_aliases(args.config) if args.config else {}

    if args.index_only:
        print_availability(rows)
        if specs:
            print_coverage(index, specs, aliases)
        return

    if not specs:
        raise SystemExit("--config is required unless --index-only")

    opt = PipelineOptions(
        out_size=(args.out_size, args.out_size),
        norm_method=args.norm,
        pad_mode=args.pad_mode,
        crop_mode=args.crop_mode,
        mask_dilate_px=args.mask_dilate_px,
        foreground_only=not args.no_foreground_norm,
        overlay=args.overlay,
    )
    resolve = make_dataset_resolver(index, aliases)
    gt_for = make_gt_lookup(args.labels_dir, args.imaging_features_key) if args.labels_dir else None
    if args.labels_dir:
        log.info("ground-truth labels from %s (missing -> 'unknown')", args.labels_dir)

    meta_path = args.metadata or (args.out_root / "metadata.csv")
    with MetadataWriter(meta_path) as writer:
        cases = sorted(index)
        for i, case in enumerate(cases, 1):
            log.info("[%d/%d] %s", i, len(cases), case)
            process_case(case, specs, args.out_root, opt, resolve, writer, gt_for=gt_for)

    if args.clinical_csv:
        merge_clinical(meta_path, args.clinical_csv, args.clinical_key_col, args.clinical_cols)

    log.info("done -> %s", meta_path)


if __name__ == "__main__":
    main()
