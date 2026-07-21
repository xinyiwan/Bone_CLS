"""
Extract the ground-truth clinical-feature JSON (``assessment.json``) for each
included subject.

Data layout (same tree walked by seg_model/pairs.py and preprocess/run.py):

    <root>/<subject>/<session>/review/<xxx>/segs/<scan>_seg.nii(.gz)   reviewed segs
    <root>/<subject>/<session>/review/<xxx>/assessment.json            feature JSON

Selection rules:

  * The feature JSON is always named ``assessment.json``. If it exists, it is
    valid -- no content check needed.
  * A subject usually has exactly one; take it.
  * When a subject has SEVERAL ``assessment.json`` (across review folders /
    sessions), take the one whose review folder has valid reviewed
    segmentations (``segs/`` with ``*_seg.nii(.gz)``). If several -- or none --
    qualify, pick deterministically by path and flag it in the report.

Included subjects come from a CSV with a patient-id column (not every subject on
disk is used). Only those are processed.

Outputs (under --out-dir):
    jsons/<subject>.json      the selected assessment.json, copied verbatim
    selection_report.csv      one row per included subject: what was picked & why
And a summary is printed to stdout.

Usage:
    python json_extract.py --data-root /data --subjects included.csv \
        --out-dir ./label_out
    # If the id column isn't auto-detected, name it:
    python json_extract.py ... --id-col Paciente
    # Require the segs to actually contain a non-empty mask (loads nibabel):
    python json_extract.py ... --require-seg-content
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger("json_extract")

# --- on-disk conventions (kept in sync with seg_model/pairs.py) ---------------
REVIEW_DIR = "review"
SEG_DIRNAME = "segs"
SEG_SUFFIXES = ("_seg.nii.gz", "_seg.nii")
JSON_NAME = "assessment.json"

# Candidate id-column names, tried in order when --id-col is not given.
ID_COL_CANDIDATES = ("subject", "case", "patient_id", "patientid", "paciente", "id")


def review_folder_has_valid_segs(review_folder: Path, require_content: bool) -> bool:
    """True if this review/<xxx>/ has reviewed segmentations.

    Cheap default: its ``segs/`` contains at least one ``*_seg.nii(.gz)`` file.
    With require_content=True, also load each mask and require a non-zero voxel
    (needs nibabel) -- catches all-background placeholder masks.
    """
    segs_dir = review_folder / SEG_DIRNAME
    if not segs_dir.is_dir():
        return False
    seg_files = [p for p in segs_dir.iterdir()
                 if p.is_file() and p.name.endswith(SEG_SUFFIXES)]
    if not seg_files:
        return False
    if not require_content:
        return True
    import numpy as np
    import nibabel as nib
    for p in seg_files:
        try:
            if np.asanyarray(nib.load(str(p)).dataobj).any():
                return True
        except Exception as e:  # noqa: BLE001
            log.debug("could not read seg %s: %s", p, e)
    return False


# =============================================================================
# Discovery + selection
# =============================================================================
@dataclass
class JsonCandidate:
    path: Path
    review_folder: Path          # the review/<xxx> this JSON belongs to
    has_reviewed_seg: bool


@dataclass
class Selection:
    subject: str
    status: str                  # ok_single | ok_reviewed | ok_ambiguous | no_json | subject_not_found
    chosen: Optional[Path] = None
    review_folder: Optional[Path] = None
    candidates: List[JsonCandidate] = field(default_factory=list)
    note: str = ""


def _review_folder_of(json_path: Path, review_root: Path) -> Path:
    """The review/<xxx> directory a JSON belongs to (immediate child of review/,
    or review/ itself if the JSON sits directly under it)."""
    rel = json_path.relative_to(review_root)
    return review_root / rel.parts[0] if len(rel.parts) > 1 else review_root


def gather_candidates(subject_dir: Path, require_content: bool) -> List[JsonCandidate]:
    """Every assessment.json under any <session>/review/ for a subject, each
    tagged with its review folder and whether that folder has reviewed segs."""
    candidates: List[JsonCandidate] = []
    # subject/<session>/review/... -- also tolerate subject/review/... directly.
    review_roots = list(subject_dir.glob(f"*/{REVIEW_DIR}"))
    direct = subject_dir / REVIEW_DIR
    if direct.is_dir():
        review_roots.append(direct)

    seg_cache: Dict[Path, bool] = {}
    for review_root in review_roots:
        if not review_root.is_dir():
            continue
        for json_path in sorted(review_root.rglob(JSON_NAME)):
            folder = _review_folder_of(json_path, review_root)
            if folder not in seg_cache:
                seg_cache[folder] = review_folder_has_valid_segs(folder, require_content)
            candidates.append(JsonCandidate(
                path=json_path,
                review_folder=folder,
                has_reviewed_seg=seg_cache[folder],
            ))
    return candidates


def select_for_subject(subject: str, subject_dir: Path,
                       require_content: bool) -> Selection:
    if not subject_dir.is_dir():
        return Selection(subject, "subject_not_found",
                         note=f"no directory at {subject_dir}")

    cands = gather_candidates(subject_dir, require_content)
    if not cands:
        return Selection(subject, "no_json", candidates=cands,
                         note=f"no {JSON_NAME} under any review/ folder")

    # Exactly one -> take it.
    if len(cands) == 1:
        c = cands[0]
        return Selection(subject, "ok_single", c.path, c.review_folder, cands)

    # Several -> prefer the one whose review folder has reviewed segmentations.
    reviewed = [c for c in cands if c.has_reviewed_seg]
    if len(reviewed) == 1:
        c = reviewed[0]
        return Selection(subject, "ok_reviewed", c.path, c.review_folder, cands,
                         note=f"{len(cands)} candidates; picked the reviewed-seg one")
    if len(reviewed) > 1:
        c = min(reviewed, key=lambda x: str(x.path))
        return Selection(subject, "ok_ambiguous", c.path, c.review_folder, cands,
                         note=f"{len(reviewed)} candidates have reviewed segs; picked first by path")
    # Several, none with reviewed segs.
    c = min(cands, key=lambda x: str(x.path))
    return Selection(subject, "ok_ambiguous", c.path, c.review_folder, cands,
                     note=f"{len(cands)} candidates, none with reviewed segs; picked first by path")


# =============================================================================
# I/O
# =============================================================================
def read_included_subjects(csv_path: Path, id_col: Optional[str]) -> List[str]:
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        if id_col is None:
            lower = {f.lower(): f for f in fields}
            for cand in ID_COL_CANDIDATES:
                if cand in lower:
                    id_col = lower[cand]
                    break
            if id_col is None:
                raise SystemExit(
                    f"Could not auto-detect the id column in {csv_path} "
                    f"(columns: {fields}). Pass --id-col.")
            log.info("using id column %r", id_col)
        elif id_col not in fields:
            raise SystemExit(f"--id-col {id_col!r} not in {csv_path} (columns: {fields})")

        ids, seen = [], set()
        for row in reader:
            sid = (row.get(id_col) or "").strip()
            if sid and sid not in seen:
                seen.add(sid)
                ids.append(sid)
    return ids


def write_report(selections: List[Selection], out_csv: Path, data_root: Path) -> None:
    def rel(p: Optional[Path]) -> str:
        if p is None:
            return ""
        try:
            return str(p.relative_to(data_root))
        except ValueError:
            return str(p)

    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["subject", "status", "chosen_json", "review_folder",
                    "n_json", "n_with_reviewed_seg", "all_candidates", "note"])
        for s in selections:
            n_seg = sum(c.has_reviewed_seg for c in s.candidates)
            allc = " ; ".join(
                f"{rel(c.path)}[seg={int(c.has_reviewed_seg)}]" for c in s.candidates)
            w.writerow([s.subject, s.status, rel(s.chosen), rel(s.review_folder),
                        len(s.candidates), n_seg, allc, s.note])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True,
                    help="Root of the <subject>/<session>/review/... tree")
    ap.add_argument("--subjects", type=Path, required=True,
                    help="CSV of included subjects (must have a patient-id column)")
    ap.add_argument("--id-col", default=None,
                    help=f"id column name (auto-detected from {ID_COL_CANDIDATES})")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Output dir: jsons/<subject>.json + selection_report.csv")
    ap.add_argument("--require-seg-content", action="store_true",
                    help="Treat segs as valid only if a mask has a non-zero voxel (needs nibabel)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")

    subjects = read_included_subjects(args.subjects, args.id_col)
    log.info("included subjects: %d", len(subjects))

    jsons_dir = args.out_dir / "jsons"
    jsons_dir.mkdir(parents=True, exist_ok=True)

    selections: List[Selection] = []
    for sid in subjects:
        sel = select_for_subject(sid, args.data_root / sid, args.require_seg_content)
        selections.append(sel)
        if sel.chosen is not None:
            shutil.copyfile(sel.chosen, jsons_dir / f"{sid}.json")

    report = args.out_dir / "selection_report.csv"
    write_report(selections, report, args.data_root)

    # --- summary ---
    counts = Counter(s.status for s in selections)
    picked = sum(1 for s in selections if s.chosen is not None)
    print("\n=== assessment.json selection summary ===")
    for status in ("ok_single", "ok_reviewed", "ok_ambiguous", "no_json", "subject_not_found"):
        if counts.get(status):
            print(f"  {status:18s}: {counts[status]}")
    print(f"  {'-'*30}")
    print(f"  selected JSONs      : {picked}/{len(subjects)}  -> {jsons_dir}")
    print(f"  report              : {report}")
    attention = [s for s in selections if s.chosen is None or s.status == "ok_ambiguous"]
    if attention:
        print(f"\n  {len(attention)} subject(s) need attention:")
        for s in attention:
            print(f"    - {s.subject}: {s.status} ({s.note})")


if __name__ == "__main__":
    main()
