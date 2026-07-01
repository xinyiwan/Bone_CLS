"""Overview of segmentation progress, cross-checked against the actual data.

Reports how many cases are:
  - both segmented AND reviewed
  - only segmented (not yet reviewed)
  - excluded

...at both the IMAGE level (one scan/session) and the SUBJECT level, using the
segmentation record CSV, and (when a data root is given) CONFIRMING those
records against the masks actually present on disk.

CSV logic (same as to_nnunet.py load_exclusions):
  - If_segmented == 'exclude'   -> excluded
  - If_segmented == 'done'      -> segmented
  - second_review non-blank     -> reviewed (always a subset of segmented)
  - If_segmented blank          -> not segmented (ignored in totals)

Disk logic (via seg_model/pairs.py find_pairs; handles *_seg.nii and *_seg.nii.gz):
  - a reviewed mask (review/<x>/segs/<scan>_seg.nii*) -> segmented + reviewed
  - a history mask (segmentation_history/segs/<scan>_seg.nii*) -> only segmented
  - no mask -> not segmented on disk (an excluded scan has no mask)
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO_ROOT / "kira-0515-seg.csv"
DEFAULT_DATA_ROOT = Path("/home/ext_xinwan/Bone_AI/tmp_sorted_data")

# Reuse the segmentation-driven discovery so CSV and disk stay consistent.
sys.path.insert(0, str(REPO_ROOT / "working" / "seg_model"))


def session_date(s: str) -> str:
    """The 8-digit YYYYMMDD embedded in a session folder name (or '')."""
    m = re.search(r"(\d{8})", s)
    return m.group(1) if m else ""


def csv_date(row) -> str:
    """YYYYMMDD from the CSV 'fechaHoraRealizacion' field (or '')."""
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", str(row.get("fechaHoraRealizacion", "")))
    return "".join(m.groups()) if m else ""


def classify_row(row):
    """Return 'both', 'only_segmented', 'excluded', or None (blank/not segmented)."""
    seg = str(row.get("If_segmented", "")).strip().lower()
    reviewed = str(row.get("second_review", "")).strip() != ""
    if seg == "exclude":
        return "excluded"
    if seg == "done":
        return "both" if reviewed else "only_segmented"
    return None


# ---------------------------------------------------------------------------
# CSV side
# ---------------------------------------------------------------------------
def summarise_csv(csv_path):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    image_counts = {"both": 0, "only_segmented": 0, "excluded": 0}
    by_subject = defaultdict(set)
    # (subject, date) -> category, for reconciliation with disk.
    session_cat = {}

    for r in rows:
        cat = classify_row(r)
        if cat is None:
            continue
        image_counts[cat] += 1
        subj = str(r.get("subject_code", "")).strip()
        if subj:
            by_subject[subj].add(cat)
            session_cat[(subj, csv_date(r))] = cat

    subject_counts = {"both": 0, "only_segmented": 0, "excluded": 0}
    mixed_seg_excl = []
    for subj, cats in by_subject.items():
        if "both" in cats:
            subject_counts["both"] += 1
        elif "only_segmented" in cats:
            subject_counts["only_segmented"] += 1
        elif "excluded" in cats:
            subject_counts["excluded"] += 1
        if "excluded" in cats and ("both" in cats or "only_segmented" in cats):
            mixed_seg_excl.append(subj)

    return rows, image_counts, subject_counts, mixed_seg_excl, session_cat


# ---------------------------------------------------------------------------
# Disk side
# ---------------------------------------------------------------------------
def summarise_disk(data_root):
    """Walk the data tree; return counts + per-session status from actual masks."""
    from pairs import find_pairs  # imported lazily (needs nibabel/pandas)

    image_counts = {"both": 0, "only_segmented": 0}
    by_subject = defaultdict(set)
    # (subject, date) -> {'both'|'only_segmented'} present, and scan list.
    session_status = defaultdict(set)
    session_scans = defaultdict(list)

    for subject, session, scan, image_path, _seg_path, source in find_pairs(data_root):
        cat = "both" if source == "reviewed" else "only_segmented"
        image_counts[cat] += 1
        by_subject[subject].add(cat)
        key = (subject, session_date(session))
        session_status[key].add(cat)
        session_scans[key].append((scan, cat, image_path is not None))

    subject_counts = {"both": 0, "only_segmented": 0}
    for cats in by_subject.values():
        if "both" in cats:
            subject_counts["both"] += 1
        elif "only_segmented" in cats:
            subject_counts["only_segmented"] += 1

    return image_counts, subject_counts, session_status, session_scans


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------
def reconcile(csv_session_cat, disk_session_status):
    """Compare CSV records with masks on disk. Returns a dict of discrepancy lists."""
    d = {
        "recorded_segmented_no_mask": [],   # CSV done, nothing on disk
        "recorded_excluded_has_mask": [],   # CSV exclude, but a mask exists
        "recorded_reviewed_no_review": [],  # CSV both, but no reviewed mask on disk
        "mask_not_recorded": [],            # mask on disk, CSV not 'done'
    }

    for (subj, date), cat in csv_session_cat.items():
        disk = disk_session_status.get((subj, date), set())
        has_mask = bool(disk)
        has_review = "both" in disk
        if cat in ("both", "only_segmented") and not has_mask:
            d["recorded_segmented_no_mask"].append((subj, date, cat))
        if cat == "both" and has_mask and not has_review:
            d["recorded_reviewed_no_review"].append((subj, date))
        if cat == "excluded" and has_mask:
            d["recorded_excluded_has_mask"].append((subj, date))

    for (subj, date), disk in disk_session_status.items():
        cat = csv_session_cat.get((subj, date))
        if cat not in ("both", "only_segmented"):
            d["mask_not_recorded"].append((subj, date, cat or "not-in-csv"))

    return d


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def print_table(title, counts, include_excluded=True):
    print(f"\n{title}")
    print("-" * len(title))
    print(f"  both segmented & reviewed : {counts.get('both', 0)}")
    print(f"  only segmented            : {counts.get('only_segmented', 0)}")
    if include_excluded:
        print(f"  excluded                  : {counts.get('excluded', 0)}")
    print(f"  total                     : {sum(counts.values())}")


def print_discrepancies(d):
    labels = {
        "recorded_segmented_no_mask": "CSV says segmented, but NO mask on disk",
        "recorded_excluded_has_mask": "CSV says excluded, but a mask EXISTS on disk",
        "recorded_reviewed_no_review": "CSV says reviewed, but no reviewed mask on disk",
        "mask_not_recorded": "mask on disk, but NOT recorded 'done' in CSV",
    }
    print("\nReconciliation (CSV vs disk)")
    print("----------------------------")
    total = sum(len(v) for v in d.values())
    if total == 0:
        print("  ✓ CSV and disk agree on all sessions.")
        return
    for key, label in labels.items():
        items = d[key]
        print(f"\n  [{len(items)}] {label}")
        for entry in sorted(items)[:25]:
            print(f"      {entry}")
        if len(items) > 25:
            print(f"      ... (+{len(items) - 25} more)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("csv", nargs="?", type=Path, default=DEFAULT_CSV,
                    help=f"segmentation record CSV (default: {DEFAULT_CSV})")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                    help="root of the BONE_AI_* data tree to confirm against")
    ap.add_argument("--no-disk", action="store_true",
                    help="skip the disk confirmation, report CSV only")
    args = ap.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")

    rows, img_c, subj_c, mixed, session_cat = summarise_csv(args.csv)

    print(f"Segmentation overview  ({args.csv.name}, {len(rows)} rows)")
    print("\n===== FROM CSV RECORD =====")
    print_table("IMAGE level (one row per scan/session)", img_c)
    print_table("SUBJECT level (aggregated by subject_code)", subj_c)
    if mixed:
        print(f"\nNote: {len(mixed)} subject(s) have BOTH segmented and excluded "
              f"sessions (counted under segmented at subject level): "
              f"{', '.join(sorted(mixed))}")

    if args.no_disk:
        return
    if not args.data_root.exists():
        print(f"\n[disk confirmation skipped: data root not found: {args.data_root}]")
        print("  Run on the machine that has the data, or pass --data-root, "
              "or use --no-disk.")
        return

    disk_img_c, disk_subj_c, disk_status, _ = summarise_disk(args.data_root)
    print("\n===== FROM ACTUAL DATA (masks on disk) =====")
    print_table("IMAGE level (one mask per scan)", disk_img_c, include_excluded=False)
    print_table("SUBJECT level", disk_subj_c, include_excluded=False)

    print_discrepancies(reconcile(session_cat, disk_status))


if __name__ == "__main__":
    main()
