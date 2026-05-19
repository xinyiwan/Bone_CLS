"""
Performance analysis of the DICOM sequence classifier.

The classifier produces three outputs per series:
    - Predicción Clases W   -> weighting   (T1W | T2W | Other | DW)
    - Predicción Clases FS  -> fat-sat     (Y | N | -)
    - Predicción Clases C   -> contrast    (Y | N | -)

Reviewed ground-truth lives in:
    - Clase W Final / Clase FS Final / Clase C Final

Pre-processing:
    0. Both input CSVs have 'Paciente' and 'Serie' swapped in their headers.
       This script swaps them back so that 'Paciente' = subject ID
       (e.g. BONE_AI_706) and 'Serie' = series name (e.g. 6_SAGT2FATLCA).
    1. The two CSVs overlap. They are concatenated, de-duplicated on the
       'Nombre DICOM' path and sorted by 'Paciente'.
    2. Per-classifier confusion matrices, per-class precision/recall/F1
       and overall accuracy are computed against the Final columns.

Out-of-classifier labels in the W truth (T2*, PD, Localizer, Zip/JPG) are 
reported in the raw confusion matrix and additionally excluded in a
'restricted' analysis so per-class metrics on the reachable labels are not
artificially depressed.

For the FS classifier, 'Y-STIR' is folded into 'Y' (STIR is fat suppression).

Usage:
    python clf_performance_analysis.py \
        /Users/xinyi/Documents/github/Bone_CLS/Review_Sequence_Classifier.csv \
        /Users/xinyi/Documents/github/Bone_CLS/Review_Sequence_Classifier_n.csv \
        --out-dir /Users/xinyi/Documents/github/Bone_CLS/working/analysis/clf_perf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Step 0 + 1: load, fix column swap, combine, dedupe, sort
# ---------------------------------------------------------------------------

PRED_COLS = ["Predicción Clases W", "Predicción Clases FS", "Predicción Clases C"]
TRUTH_COLS = ["Clase W Final",      "Clase FS Final",       "Clase C Final"]

# Label sets the classifier can output
W_CLASSES  = ["T1W", "T2W", "Other", "DW"]
FS_CLASSES = ["Y", "N", "-"]
C_CLASSES  = ["Y", "N", "-"]

# W truth labels not in W_CLASSES — kept in raw matrix, excluded from restricted
# Need to check if there are other out-of-class labels in FS/C, but W is the one with known issues from review
W_OUT_OF_CLASS = {"T2*", "PD", "Localizer", "Zip/JPG"}


def load_and_fix(path: Path) -> pd.DataFrame:
    """Load a review CSV and swap the mislabelled Paciente / Serie columns."""
    df = pd.read_csv(path, low_memory=False)
    if not {"Paciente", "Serie"}.issubset(df.columns):
        raise ValueError(f"{path}: expected columns 'Paciente' and 'Serie'")
    df = df.rename(columns={"Paciente": "Serie", "Serie": "Paciente"})
    df["__source"] = path.name
    return df


def combine(csv_a: Path, csv_b: Path) -> pd.DataFrame:
    a = load_and_fix(csv_a)
    b = load_and_fix(csv_b)
    print(f"Loaded {csv_a.name}: {len(a):,} rows")
    print(f"Loaded {csv_b.name}: {len(b):,} rows")

    combined = pd.concat([a, b], ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["Nombre DICOM"], keep="first")
    after = len(combined)
    print(f"Combined: {before:,} -> {after:,} after dedupe on 'Nombre DICOM' "
          f"({before - after:,} duplicates removed)")

    combined = combined.sort_values(["Paciente", "Estudio", "Serie"],
                                    kind="mergesort").reset_index(drop=True)
    return combined


# ---------------------------------------------------------------------------
# Step 2: per-classifier performance
# ---------------------------------------------------------------------------

def normalise_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def confusion_table(truth: pd.Series, pred: pd.Series,
                    truth_order: list[str], pred_order: list[str]) -> pd.DataFrame:
    cm = pd.crosstab(truth, pred, dropna=False)
    # Keep a stable ordering with any extra labels appended at the end
    truth_extra = [l for l in cm.index    if l not in truth_order]
    pred_extra  = [l for l in cm.columns  if l not in pred_order]
    cm = cm.reindex(index=truth_order + truth_extra,
                    columns=pred_order + pred_extra,
                    fill_value=0)
    return cm


def per_class_metrics(cm: pd.DataFrame) -> pd.DataFrame:
    """Compute precision / recall / F1 / support from a (truth x pred) matrix."""
    labels = sorted(set(cm.index) | set(cm.columns))
    rows = []
    for lbl in labels:
        tp = cm.loc[lbl, lbl] if (lbl in cm.index and lbl in cm.columns) else 0
        col_sum = cm[lbl].sum()  if lbl in cm.columns else 0
        row_sum = cm.loc[lbl].sum() if lbl in cm.index else 0
        fp = col_sum - tp
        fn = row_sum - tp
        support = row_sum
        precision = tp / (tp + fp) if (tp + fp) else np.nan
        recall    = tp / (tp + fn) if (tp + fn) else np.nan
        f1 = (2 * precision * recall / (precision + recall)
              if precision and recall and (precision + recall) else np.nan)
        rows.append(dict(label=lbl, support=int(support),
                         precision=precision, recall=recall, f1=f1,
                         tp=int(tp), fp=int(fp), fn=int(fn)))
    out = pd.DataFrame(rows).set_index("label")
    return out


def overall_accuracy(truth: pd.Series, pred: pd.Series) -> float:
    return float((truth == pred).sum() / len(truth)) if len(truth) else float("nan")


def plot_confusion(cm: pd.DataFrame, title: str, out_path: Path) -> None:
    """Heatmap of the confusion matrix annotated with count + row-pct."""
    row_totals = cm.sum(axis=1).replace(0, np.nan)
    annot = cm.apply(
        lambda r: r.map(lambda v: f"{v}\n({v/row_totals[r.name]*100:.0f}%)"
                                if row_totals[r.name] and not np.isnan(row_totals[r.name])
                                else str(v)),
        axis=1,
    )
    fig, ax = plt.subplots(figsize=(max(5, len(cm.columns) * 1.2),
                                    max(4, len(cm.index)   * 0.7)))
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues",
                linewidths=0.5, linecolor="grey",
                cbar_kws={"label": "count"}, ax=ax)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Truth")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  heatmap -> {out_path}")


# ---------------------------------------------------------------------------
# Per-classifier analyses
# ---------------------------------------------------------------------------

def analyse_w(df: pd.DataFrame, out_dir: Path) -> dict:
    truth = normalise_str(df["Clase W Final"])
    pred  = normalise_str(df["Predicción Clases W"])

    # Raw confusion matrix (keeps out-of-class truths)
    cm_raw = confusion_table(truth, pred, truth_order=W_CLASSES, pred_order=W_CLASSES)
    plot_confusion(cm_raw, "W classifier — raw (incl. out-of-class truths)",
                   out_dir / "cm_W_raw.png")

    # Restricted: drop rows whose truth is outside the classifier's label set
    keep = ~truth.isin(W_OUT_OF_CLASS)
    cm_res = confusion_table(truth[keep], pred[keep],
                             truth_order=W_CLASSES, pred_order=W_CLASSES)
    plot_confusion(cm_res, "W classifier — restricted to reachable truth labels",
                   out_dir / "cm_W_restricted.png")

    metrics = per_class_metrics(cm_res)
    acc_raw = overall_accuracy(truth, pred)
    acc_res = overall_accuracy(truth[keep], pred[keep])

    print("\n[W classifier]")
    print(f"  raw accuracy        : {acc_raw:.3f}  (n={len(truth)})")
    print(f"  restricted accuracy : {acc_res:.3f}  (n={int(keep.sum())}, "
          f"{int((~keep).sum())} out-of-class truths excluded)")
    print(f"  raw confusion matrix:\n{cm_raw}\n")
    print(f"  per-class metrics (restricted):\n{metrics.round(3)}")

    return dict(cm_raw=cm_raw, cm_restricted=cm_res, metrics=metrics,
                accuracy_raw=acc_raw, accuracy_restricted=acc_res,
                n_raw=len(truth), n_restricted=int(keep.sum()))


def analyse_fs(df: pd.DataFrame, out_dir: Path) -> dict:
    truth = normalise_str(df["Clase FS Final"]).replace({"Y-STIR": "Y"})
    pred  = normalise_str(df["Predicción Clases FS"])

    cm = confusion_table(truth, pred, truth_order=FS_CLASSES, pred_order=FS_CLASSES)
    plot_confusion(cm, "FS classifier (Y-STIR folded into Y)",
                   out_dir / "cm_FS.png")

    metrics = per_class_metrics(cm)
    acc = overall_accuracy(truth, pred)

    print("\n[FS classifier]")
    print(f"  accuracy : {acc:.3f}  (n={len(truth)})")
    print(f"  confusion matrix:\n{cm}\n")
    print(f"  per-class metrics:\n{metrics.round(3)}")

    return dict(cm=cm, metrics=metrics, accuracy=acc, n=len(truth))


def analyse_c(df: pd.DataFrame, out_dir: Path) -> dict:
    truth = normalise_str(df["Clase C Final"])
    pred  = normalise_str(df["Predicción Clases C"])

    cm = confusion_table(truth, pred, truth_order=C_CLASSES, pred_order=C_CLASSES)
    plot_confusion(cm, "C classifier (contrast)",
                   out_dir / "cm_C.png")

    metrics = per_class_metrics(cm)
    acc = overall_accuracy(truth, pred)

    print("\n[C classifier]")
    print(f"  accuracy : {acc:.3f}  (n={len(truth)})")
    print(f"  confusion matrix:\n{cm}\n")
    print(f"  per-class metrics:\n{metrics.round(3)}")

    return dict(cm=cm, metrics=metrics, accuracy=acc, n=len(truth))


# ---------------------------------------------------------------------------
# Misclassified rows export
# ---------------------------------------------------------------------------

def save_errors(df: pd.DataFrame, out_dir: Path) -> None:
    key_cols = ["Paciente", "Estudio", "Serie", "Nombre DICOM"]
    truth_w  = normalise_str(df["Clase W Final"])
    truth_fs = normalise_str(df["Clase FS Final"]).replace({"Y-STIR": "Y"})
    truth_c  = normalise_str(df["Clase C Final"])
    pred_w   = normalise_str(df["Predicción Clases W"])
    pred_fs  = normalise_str(df["Predicción Clases FS"])
    pred_c   = normalise_str(df["Predicción Clases C"])

    err_mask = (truth_w != pred_w) | (truth_fs != pred_fs) | (truth_c != pred_c)
    errors = df.loc[err_mask, key_cols + PRED_COLS + TRUTH_COLS].copy()
    errors["W_wrong"]  = (truth_w  != pred_w ).loc[err_mask].values
    errors["FS_wrong"] = (truth_fs != pred_fs).loc[err_mask].values
    errors["C_wrong"]  = (truth_c  != pred_c ).loc[err_mask].values

    path = out_dir / "misclassified_rows.csv"
    errors.to_csv(path, index=False)
    print(f"\nMisclassified rows ({len(errors):,}/{len(df):,}) -> {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("csv_a", type=Path,
                        help="First review CSV (Paciente/Serie are swapped in header)")
    parser.add_argument("csv_b", type=Path,
                        help="Second review CSV (Paciente/Serie are swapped in header)")
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parent / "clf_perf",
                        help="Where to write plots and CSVs (default: ./clf_perf)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="white")

    df = combine(args.csv_a, args.csv_b)
    combined_path = args.out_dir / "combined_reviewed.csv"
    df.to_csv(combined_path, index=False)
    print(f"Combined+sorted table -> {combined_path}\n")

    w_res  = analyse_w(df,  args.out_dir)
    fs_res = analyse_fs(df, args.out_dir)
    c_res  = analyse_c(df,  args.out_dir)

    # One-line summary table
    summary = pd.DataFrame([
        dict(classifier="W (restricted)", n=w_res["n_restricted"],
             accuracy=w_res["accuracy_restricted"]),
        dict(classifier="W (raw)",        n=w_res["n_raw"],
             accuracy=w_res["accuracy_raw"]),
        dict(classifier="FS",             n=fs_res["n"],
             accuracy=fs_res["accuracy"]),
        dict(classifier="C",              n=c_res["n"],
             accuracy=c_res["accuracy"]),
    ])
    summary_path = args.out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary:\n{summary.round(3).to_string(index=False)}")
    print(f"Summary saved -> {summary_path}")

    # Per-classifier per-class metrics CSVs
    w_res["metrics"].to_csv(args.out_dir / "metrics_W.csv")
    fs_res["metrics"].to_csv(args.out_dir / "metrics_FS.csv")
    c_res["metrics"].to_csv(args.out_dir / "metrics_C.csv")
    w_res["cm_raw"].to_csv(args.out_dir / "cm_W_raw.csv")
    w_res["cm_restricted"].to_csv(args.out_dir / "cm_W_restricted.csv")
    fs_res["cm"].to_csv(args.out_dir / "cm_FS.csv")
    c_res["cm"].to_csv(args.out_dir / "cm_C.csv")

    save_errors(df, args.out_dir)


if __name__ == "__main__":
    main()
