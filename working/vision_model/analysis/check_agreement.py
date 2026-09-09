#!/usr/bin/env python3
"""Compare parsed_label agreement between single-slice and stack results."""
import sys
import pandas as pd
from sklearn.metrics import cohen_kappa_score

SINGLE = "BONE-AI/freetext/rank/freetext_slice_all_pilot40.csv"
STACK = "BONE-AI/stack/rank/freetext_slice_all_pilot40.csv"

KEYS = ["case_id", "feature_name", "plane", "modality"]


def load(path):
    df = pd.read_csv(path)
    missing = [c for c in KEYS + ["parsed_label", "ground_truth_label"] if c not in df.columns]
    if missing:
        sys.exit(f"{path}: missing columns {missing}")
    return df


def main():
    single = load(SINGLE)
    stack = load(STACK)

    merged = single.merge(stack, on=KEYS, suffixes=("_single", "_stack"), how="inner")
    n_single, n_stack, n_merged = len(single), len(stack), len(merged)
    print(f"single rows: {n_single}, stack rows: {n_stack}, matched rows: {n_merged}")
    if n_merged < min(n_single, n_stack):
        print(f"WARNING: {min(n_single, n_stack) - n_merged} rows did not match on {KEYS}")

    merged["agree"] = merged["parsed_label_single"] == merged["parsed_label_stack"]
    overall_agreement = merged["agree"].mean()
    print(f"\nOverall parsed_label agreement: {overall_agreement:.3f} ({merged['agree'].sum()}/{n_merged})")

    labels = sorted(set(merged["parsed_label_single"].dropna()) | set(merged["parsed_label_stack"].dropna()))
    valid = merged.dropna(subset=["parsed_label_single", "parsed_label_stack"])
    if len(valid) > 1 and len(labels) > 1:
        kappa = cohen_kappa_score(valid["parsed_label_single"], valid["parsed_label_stack"], labels=labels)
        print(f"Cohen's kappa: {kappa:.3f}")

    print("\nConfusion matrix (rows=single, cols=stack):")
    ct = pd.crosstab(merged["parsed_label_single"], merged["parsed_label_stack"], dropna=False)
    print(ct)

    print("\nAgreement by feature_name:")
    by_feature = merged.groupby("feature_name")["agree"].agg(["mean", "count"])
    print(by_feature.sort_values("mean"))

    print("\nAgreement by plane:")
    by_plane = merged.groupby("plane")["agree"].agg(["mean", "count"])
    print(by_plane.sort_values("mean"))

    # Correctness vs ground truth for each, where available
    for suffix in ["single", "stack"]:
        gt_col = f"ground_truth_label_{suffix}"
        pl_col = f"parsed_label_{suffix}"
        sub = merged.dropna(subset=[gt_col, pl_col])
        if len(sub):
            acc = (sub[pl_col] == sub[gt_col]).mean()
            print(f"\n{suffix} accuracy vs ground_truth_label: {acc:.3f} ({len(sub)} rows with GT)")

    out_path = "BONE-AI/agreement_single_vs_stack.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nFull merged comparison written to {out_path}")


if __name__ == "__main__":
    main()
