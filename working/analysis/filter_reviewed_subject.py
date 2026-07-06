"""Filter already-reviewed subjects out of the batch 3 / batch 4 classifier output.

Reviewed subjects come from ``combined_reviewed.csv`` (the reviewed rows of
batch 1 + batch 2). The patient id there lives in the ``Paciente`` column
(values like ``BONE_AI_296``).

We drop every row whose subject is already reviewed from each batch's
``Sequence_Classifier.csv`` file, saving the remaining (not-yet-reviewed)
subjects to a ``Sequence_Classifier_filtered.csv`` next to each input (one
output file per batch, not combined).

NOTE on patient id: the raw ``Review_Sequence_Classifier*.csv`` files have their
``Paciente`` / ``Serie`` columns swapped (the ``BONE_AI_*`` id sits in ``Serie``),
so we identify the subject from the ``Nombre DICOM`` path where possible and fall
back to whichever column actually holds a ``BONE_AI_*`` value. This keeps the
matching correct regardless of column ordering in each file.
"""

import argparse
import re
from pathlib import Path

import pandas as pd

# Defaults follow the container layout used by the other analysis scripts
# (/data for inputs, /output for results). Override on the command line.
DEFAULT_REVIEWED = Path("/output/clf_perf/combined_reviewed.csv")
DEFAULT_BATCHES = [
    Path("/data/batch_3/Results/Sequence_Classifier.csv"),
    Path("/data/batch_4/Results/Sequence_Classifier.csv"),
]
PID_RE = re.compile(r"(BONE_AI_\d+)")


def patient_ids(df: pd.DataFrame) -> pd.Series:
    """Best-effort BONE_AI_* id per row.

    Preference: the id embedded in the ``Nombre DICOM`` path; then whichever of
    ``Serie`` / ``Paciente`` contains a BONE_AI_* value (columns are sometimes
    swapped between files).
    """
    ids = pd.Series(pd.NA, index=df.index, dtype="object")
    if "Nombre DICOM" in df.columns:
        ids = df["Nombre DICOM"].astype(str).str.extract(PID_RE, expand=False)
    for col in ("Serie", "Paciente"):
        if col in df.columns:
            fallback = df[col].astype(str).str.extract(PID_RE, expand=False)
            ids = ids.fillna(fallback)
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reviewed", type=Path, default=DEFAULT_REVIEWED,
                    help=f"combined reviewed CSV (default: {DEFAULT_REVIEWED})")
    ap.add_argument("--batch", type=Path, nargs="+", default=DEFAULT_BATCHES,
                    help="batch Sequence_Classifier.csv file(s) to filter")
    args = ap.parse_args()

    if not args.reviewed.exists():
        raise SystemExit(f"Reviewed CSV not found: {args.reviewed}")
    batches = [p for p in args.batch if p.exists()]
    for p in args.batch:
        if not p.exists():
            print(f"WARNING: batch file not found, skipping: {p}")
    if not batches:
        raise SystemExit("No batch files found to filter.")

    # Reviewed subjects (batch 1 + 2). Use the clean Paciente column, but fall
    # back to the robust extractor if needed.
    rev = pd.read_csv(args.reviewed)
    reviewed = set(patient_ids(rev).dropna())
    print(f"Reviewed subjects (from {args.reviewed.name}): {len(reviewed)}\n")

    # Filter each batch independently; write one output next to each input.
    for p in batches:
        b = pd.read_csv(p)
        subject = patient_ids(b)

        no_id = int(subject.isna().sum())
        batch_subjects = set(subject.dropna())
        already = batch_subjects & reviewed
        new_subjects = batch_subjects - reviewed

        # Keep rows whose subject is NOT already reviewed (no-id rows are kept).
        keep_mask = ~subject.isin(reviewed)
        filtered = b[keep_mask]

        out = p.with_name(f"{p.stem}_filtered.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        filtered.to_csv(out, index=False)

        print(f"{p}")
        print(f"  subjects: {len(batch_subjects)} total, "
              f"{len(already)} already reviewed (dropped), "
              f"{len(new_subjects)} new (kept)")
        if no_id:
            print(f"  WARNING: {no_id} rows had no BONE_AI_* id (kept as new)")
        print(f"  rows: {len(b)} -> {len(filtered)} kept "
              f"({len(b) - len(filtered)} dropped)")
        print(f"  saved -> {out}\n")


if __name__ == "__main__":
    main()
