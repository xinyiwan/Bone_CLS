# Extract clinical information from both DICOM header and LLM results.
#
# DCM header CSVs (one per batch) contain per-scan rows with at least:
#     subject, birthdate, age, gender
# We collapse to one row per subject (first occurrence) since demographics
# are stable across a subject's scans.
#
# LLM CSV contains one row per subject (column ``pid``) with:
#     pid, symptoms, history_of_neoplasm, suspected_metastasis,
#     skeletal_location, location_within_bone
#
# The combined output has one row per subject with demographics + LLM fields.

from __future__ import annotations

import pandas as pd

llm_cli_csv = "/home/ext_xinwan/Bone_AI/BTRecordsLLM/output/clinical_variables/kira-0515-llm-cli.csv"
# dcm_batch_1_csv = "/home/ext_xinwan/Bone_AI/output/DCM_Physics/batch_1/dicom_headers_labelled_Mar24.csv"
# dcm_batch_2_csv = "/home/ext_xinwan/Bone_AI/output/DCM_Physics/batch_2/batch_2_headers_labelled.csv"

dcm_batch_1_csv = "/home/ext_xinwan/Bone_AI/output/DCM_Physics/all_batches/batch_1_labelled.csv"
dcm_batch_2_csv = "/home/ext_xinwan/Bone_AI/output/DCM_Physics/all_batches/batch_2_labelled.csv"
dcm_batch_3_csv = "/home/ext_xinwan/Bone_AI/output/DCM_Physics/all_batches/batch_3_labelled.csv"
dcm_batch_4_csv = "/home/ext_xinwan/Bone_AI/output/DCM_Physics/all_batches/batch_4_labelled.csv"

output_csv = "/home/ext_xinwan/Bone_AI/output/clinical_info/combined_clinical_info_batches1_to_4.csv"

DCM_KEEP_COLS = ["subject", "birthdate", "age", "gender"]
LLM_KEEP_COLS = [
    "pid",
    "symptoms",
    "history_of_neoplasm",
    "suspected_metastasis",
    "skeletal_location",
    "location_within_bone",
]


def load_dcm_demographics(csv_path):
    # type: (str) -> pd.DataFrame
    """Load one DCM-header CSV and return one row per subject."""
    df = pd.read_csv(csv_path, usecols=DCM_KEEP_COLS)
    df = df.drop_duplicates(subset="subject", keep="first")
    return df


def combine():
    # type: () -> pd.DataFrame
    dcm = pd.concat(
        [load_dcm_demographics(dcm_batch_1_csv), load_dcm_demographics(dcm_batch_2_csv), load_dcm_demographics(dcm_batch_3_csv), load_dcm_demographics(dcm_batch_4_csv)],
        ignore_index=True,
    )
    dcm = dcm.drop_duplicates(subset="subject", keep="first")

    llm = pd.read_csv(llm_cli_csv, usecols=LLM_KEEP_COLS)

    merged = dcm.merge(llm, left_on="subject", right_on="pid", how="outer")
    # Prefer subject id from DCM, fall back to pid for LLM-only subjects
    merged["subject"] = merged["subject"].fillna(merged["pid"])
    merged = merged.drop(columns="pid")

    return merged


if __name__ == "__main__":
    combined = combine()
    combined.to_csv(output_csv, index=False)
    print("Wrote %d subjects to %s" % (len(combined), output_csv))
