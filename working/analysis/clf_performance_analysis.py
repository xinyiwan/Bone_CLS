"""
Cascade performance analysis of the DICOM sequence classifier.

The model is a conditional cascade of three heads:

    W  (weighting)     : T1W | T2W | DW | Other        — runs on all rows
    FS (fat suppress.) : Y | N                          — runs ONLY when W in {T1W, T2W}
    C  (contrast)      : Y | N                          — runs ONLY when W == T1W

Each head is evaluated on the slice of data its working logic actually
processes, against the human-reviewed Final columns.

Pre-processing:
    0. Both input CSVs have 'Paciente' and 'Serie' swapped in their headers.
       This script swaps them back so 'Paciente' = subject ID (e.g.
       BONE_AI_706) and 'Serie' = series name (e.g. 6_SAGT2FATLCA).
    1. The two CSVs overlap. They are concatenated, de-duplicated on the
       'Nombre DICOM' path and sorted by 'Paciente'.
    2. Rows with W truth in {Other, Localizer, Zip/JPG} are dropped — that
       'Other' bucket lumps heterogeneous sequences (e.g. perfusion that may
       actually be T1) and biases scoring. 'Y-STIR' in FS truth is folded
       into 'Y'.

Analyses (each head evaluated on the rows eligible at the previous stage —
i.e. cascade-conditional):
    * W classifier — basic metrics (acc, F1, precision, sensitivity, AUC)
      on truth in {T1W, T2W, DW}; plus a full confusion matrix that ALSO
      shows how T2* / PD truths map to the W predictions.
    * FS classifier — restricted to rows where truth W ∈ {T1W, T2W, DW}
      AND the W head predicted T1W or T2W (the FS head's input domain).
    * C classifier — restricted to rows where truth W ∈ {T1W, T2W, DW}
      AND the W head predicted T1W (the C head's input domain).
    * Composite (final cascade label) — T1W_(n)FS_(n)CE | T2W_(n)FS.
      For T2W, the C dimension is collapsed (T2W_FS_N == T2W_FS_-).
      DWI, T2*, PD truths are excluded from the composite analysis since
      they are not target labels.

AUC: two modes, auto-detected per head.

  * If the input CSV contains a JSON-encoded per-class probability column
    (named in PROBAS_COLS, e.g. 'Predicción Clases W Probas' with values
    like '{"T1W": 0.95, "T2W": 0.03, "DW": 0.01, "Other": 0.01}'), the
    script computes a proper per-class one-vs-rest AUC and reports the
    macro average over the head's target classes.
  * If that column is absent, the script falls back to a single
    *confidence-AUC* per head:
        y     = (truth == pred)
        score = p   (the model's top-class probability)
    This measures whether the model's stated confidence ranks correct
    predictions above incorrect ones.

To unlock proper macro-OvR AUC, modify the classifier's inference loop to
save the full predict_proba() vector per row rather than just np.max(...).

Usage:
    python clf_performance_analysis.py \
        /Users/xinyi/Documents/github/Bone_CLS/Review_Sequence_Classifier.csv \
        /Users/xinyi/Documents/github/Bone_CLS/Review_Sequence_Classifier_n.csv \
        --out-dir /Users/xinyi/Documents/github/Bone_CLS/working/analysis/clf_perf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRED_COLS  = ["Predicción Clases W",   "Predicción Clases FS",   "Predicción Clases C"]
PROB_COLS  = ["Predicción Clases W P", "Predicción Clases FS P", "Predicción Clases C P"]
TRUTH_COLS = ["Clase W Final",         "Clase FS Final",         "Clase C Final"]

# Optional JSON-encoded per-class probability columns. When present, proper
# macro-OvR AUC is computed; otherwise we fall back to confidence-AUC.
# Expected format per row: '{"T1W": 0.95, "T2W": 0.03, "DW": 0.01, "Other": 0.01}'
PROBAS_COLS = {
    "W":  "Predicción Clases W Probas",
    "FS": "Predicción Clases FS Probas",
    "C":  "Predicción Clases C Probas",
}

# W truth labels dropped globally in step 1 (heterogeneous / non-target)
W_TRUTH_DROP = {"Other", "Localizer", "Zip/JPG"}

# W classes that proceed to the FS head, and to the C head
FS_INPUT_W = {"T1W", "T2W"}
C_INPUT_W  = {"T1W"}

# Cascade composite target labels
FINAL_LABELS = [
    "T1W_nFS_nCE", "T1W_nFS_CE", "T1W_FS_nCE", "T1W_FS_CE",
    "T2W_nFS",     "T2W_FS",
]

# W truth labels considered reachable by the W head — the cascade-eligible set.
# T2*, PD, etc. are out-of-class for the W head, so we exclude them from every
# downstream head's evaluation as well (FS and C are conditional on W).
W_IN_CLASS = {"T1W", "T2W", "DW"}


# ---------------------------------------------------------------------------
# Step 0 + 1: load, fix column swap, combine, dedupe, sort
# ---------------------------------------------------------------------------

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
# Shared helpers
# ---------------------------------------------------------------------------

def normalise_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def fs_truth(df: pd.DataFrame) -> pd.Series:
    """Normalised FS truth with Y-STIR folded into Y."""
    return normalise_str(df["Clase FS Final"]).replace({"Y-STIR": "Y"})


def confusion_table(truth: pd.Series, pred: pd.Series,
                    truth_order: list[str], pred_order: list[str]) -> pd.DataFrame:
    cm = pd.crosstab(truth, pred, dropna=False)
    truth_extra = [l for l in cm.index   if l not in truth_order]
    pred_extra  = [l for l in cm.columns if l not in pred_order]
    cm = cm.reindex(index=truth_order + truth_extra,
                    columns=pred_order + pred_extra,
                    fill_value=0)
    return cm


def per_class_metrics(cm: pd.DataFrame) -> pd.DataFrame:
    """Per-label precision / recall / F1 / support from a (truth x pred) matrix."""
    labels = list(dict.fromkeys(list(cm.index) + list(cm.columns)))
    rows = []
    for lbl in labels:
        tp = cm.loc[lbl, lbl] if (lbl in cm.index and lbl in cm.columns) else 0
        col_sum = cm[lbl].sum()       if lbl in cm.columns else 0
        row_sum = cm.loc[lbl].sum()   if lbl in cm.index   else 0
        fp = col_sum - tp
        fn = row_sum - tp
        precision = tp / (tp + fp) if (tp + fp) else np.nan
        recall    = tp / (tp + fn) if (tp + fn) else np.nan
        f1 = (2 * precision * recall / (precision + recall)
              if precision and recall and (precision + recall) else np.nan)
        rows.append(dict(label=lbl, support=int(row_sum),
                         precision=precision, recall=recall, f1=f1,
                         tp=int(tp), fp=int(fp), fn=int(fn)))
    return pd.DataFrame(rows).set_index("label")


def overall_accuracy(truth: pd.Series, pred: pd.Series) -> float:
    return float((truth == pred).sum() / len(truth)) if len(truth) else float("nan")


def plot_confusion(cm: pd.DataFrame, title: str, out_path: Path) -> None:
    """Heatmap of the confusion matrix annotated with count + row-percent."""
    row_totals = cm.sum(axis=1).replace(0, np.nan)
    annot = cm.apply(
        lambda r: r.map(lambda v: f"{v}\n({v/row_totals[r.name]*100:.0f}%)"
                                if row_totals[r.name] and not np.isnan(row_totals[r.name])
                                else str(v)),
        axis=1,
    )
    fig, ax = plt.subplots(figsize=(max(5, len(cm.columns) * 1.3),
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
# AUC helper (top-class probability only)
# ---------------------------------------------------------------------------
#
# We only have the probability of the top predicted class, so the only
# meaningful AUC we can compute is the *confidence-AUC*:
#     y     = (truth == pred)   # was the prediction correct?
#     score = p                 # the model's stated top-class probability

def _safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    mask = ~np.isnan(score)
    if mask.sum() < 2 or len(np.unique(y_true[mask])) < 2:
        return float("nan")
    return float(roc_auc_score(y_true[mask], score[mask]))


def parse_probas(df: pd.DataFrame, col: str) -> pd.DataFrame | None:
    """
    Parse a JSON-encoded per-class probability column into a DataFrame whose
    columns are class names. Returns None if the column is missing, empty,
    or unparseable for every row.
    """
    if col not in df.columns:
        return None
    def _loads(s):
        if isinstance(s, str) and s.strip():
            try:
                obj = json.loads(s)
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                return None
        return None
    parsed = df[col].apply(_loads)
    if parsed.map(lambda d: isinstance(d, dict)).sum() == 0:
        return None
    classes = sorted({k for d in parsed if isinstance(d, dict) for k in d.keys()})
    rows = [
        [float(d.get(c, np.nan)) for c in classes] if isinstance(d, dict)
        else [np.nan] * len(classes)
        for d in parsed
    ]
    return pd.DataFrame(rows, columns=classes, index=df.index)


# ---------------------------------------------------------------------------
# Generic "evaluate one head" routine
# ---------------------------------------------------------------------------

def evaluate_head(name: str,
                  truth: pd.Series, pred: pd.Series, prob: pd.Series,
                  classes: list[str], out_dir: Path,
                  proba_matrix: pd.DataFrame | None = None) -> dict:
    """
    Compute accuracy, per-class precision/recall/F1/support, confidence-AUC,
    and — if `proba_matrix` (per-class probabilities) is provided — proper
    per-class one-vs-rest AUC plus macro-OvR AUC.
    Save confusion-matrix heatmap, metrics CSV, and the appropriate ROC plot.
    `classes` defines the row/column ordering for the confusion matrix.
    """
    cm = confusion_table(truth, pred, truth_order=classes, pred_order=classes)
    metrics = per_class_metrics(cm)

    acc = overall_accuracy(truth, pred)
    correct = (truth.values == pred.values).astype(int)
    conf_auc = _safe_auc(correct, prob.values)

    target_metrics = metrics.reindex(classes)
    macro_prec = float(target_metrics["precision"].mean(skipna=True))
    macro_rec  = float(target_metrics["recall"].mean(skipna=True))
    macro_f1   = float(target_metrics["f1"].mean(skipna=True))

    # Proper per-class OvR AUC (requires per-class probabilities)
    per_class_auc: dict[str, float] = {}
    macro_ovr_auc = float("nan")
    if proba_matrix is not None:
        for cls in classes:
            if cls in proba_matrix.columns:
                y = (truth.values == cls).astype(int)
                s = proba_matrix[cls].values
                per_class_auc[cls] = _safe_auc(y, s)
            else:
                per_class_auc[cls] = float("nan")
        finite = [v for v in per_class_auc.values() if not np.isnan(v)]
        if finite:
            macro_ovr_auc = float(np.mean(finite))
        metrics["auc"] = metrics.index.map(per_class_auc)

    print(f"\n[{name}]  n={len(truth):,}")
    print(f"  accuracy={acc:.3f}   "
          f"macro-precision={macro_prec:.3f}  "
          f"macro-sensitivity(recall)={macro_rec:.3f}  "
          f"macro-F1={macro_f1:.3f}")
    if proba_matrix is not None:
        print(f"  per-class AUC: " +
              ", ".join(f"{c}={per_class_auc[c]:.3f}" for c in classes))
        print(f"  macro one-vs-rest AUC = {macro_ovr_auc:.3f}    "
              f"confidence-AUC = {conf_auc:.3f}")
    else:
        print(f"  confidence-AUC = {conf_auc:.3f}    "
              f"(per-class AUC unavailable — no '{PROBAS_COLS.get(name, '?')}' "
              f"column in input)")
    print(f"  confusion matrix:\n{cm}")
    print(f"  per-class metrics:\n{metrics.round(3)}")

    plot_confusion(cm, f"{name} — confusion matrix",
                   out_dir / f"cm_{name}.png")
    cm.to_csv(out_dir / f"cm_{name}.csv")
    metrics.to_csv(out_dir / f"metrics_{name}.csv")

    # ROC plot: per-class OvR if probas available, otherwise confidence ROC
    if proba_matrix is not None:
        fig, ax = plt.subplots(figsize=(5.5, 5))
        plotted = False
        for cls in classes:
            if cls not in proba_matrix.columns:
                continue
            y = (truth.values == cls).astype(int)
            s = proba_matrix[cls].values
            mask = ~np.isnan(s)
            if mask.sum() < 2 or len(np.unique(y[mask])) < 2:
                continue
            fpr, tpr, _ = roc_curve(y[mask], s[mask])
            ax.plot(fpr, tpr,
                    label=f"{cls}  AUC={per_class_auc[cls]:.3f}  n+={int(y.sum())}")
            plotted = True
        if plotted:
            ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=0.8)
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.set_title(f"{name} — one-vs-rest ROC  (macro AUC = {macro_ovr_auc:.3f})")
            ax.legend(loc="lower right", fontsize=8)
            plt.tight_layout()
            fig.savefig(out_dir / f"roc_{name}.png", dpi=150)
        plt.close(fig)
    elif len(np.unique(correct)) == 2 and not np.isnan(prob.values).all():
        fpr, tpr, _ = roc_curve(correct, prob.values)
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.plot(fpr, tpr, label=f"AUC = {conf_auc:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=0.8)
        ax.set_xlabel("FPR (incorrect ranked above correct)")
        ax.set_ylabel("TPR (correct ranked above incorrect)")
        ax.set_title(f"{name} — confidence ROC")
        ax.legend(loc="lower right")
        plt.tight_layout()
        fig.savefig(out_dir / f"roc_{name}.png", dpi=150)
        plt.close(fig)

    return dict(name=name, n=len(truth),
                accuracy=acc, macro_precision=macro_prec,
                macro_recall=macro_rec, macro_f1=macro_f1,
                confidence_auc=conf_auc,
                macro_ovr_auc=macro_ovr_auc,
                per_class_auc=per_class_auc,
                cm=cm, metrics=metrics)


# ---------------------------------------------------------------------------
# Step 2: W classifier (basic metrics + extended confusion matrix)
# ---------------------------------------------------------------------------

def analyse_w(df: pd.DataFrame, out_dir: Path) -> dict:
    truth_all = normalise_str(df["Clase W Final"])
    pred_all  = normalise_str(df["Predicción Clases W"])
    prob_all  = pd.to_numeric(df["Predicción Clases W P"], errors="coerce")
    probas_all = parse_probas(df, PROBAS_COLS["W"])

    # (a) Basic metrics: truth restricted to classifier's three real targets
    in_class = truth_all.isin(W_IN_CLASS)
    res = evaluate_head(
        name="W",
        truth=truth_all[in_class],
        pred=pred_all[in_class],
        prob=prob_all[in_class],
        classes=["T1W", "T2W", "DW"],
        out_dir=out_dir,
        proba_matrix=(probas_all.loc[in_class] if probas_all is not None else None),
    )
    print(f"  (basic metrics computed on n={int(in_class.sum())} rows "
          f"with truth in {{T1W, T2W, DW}}; "
          f"{int((~in_class).sum())} rows with extra truth labels held back "
          f"for the extended confusion matrix below.)")

    # (b) Full confusion matrix incl. T2* and PD truth rows
    truth_order = ["T1W", "T2W", "DW"] + sorted(
        set(truth_all.unique()) - {"T1W", "T2W", "DW"})
    pred_order  = ["T1W", "T2W", "DW", "Other"]
    cm_full = confusion_table(truth_all, pred_all,
                              truth_order=truth_order, pred_order=pred_order)
    plot_confusion(cm_full, "W classifier — extended (incl. T2*/PD truth)",
                   out_dir / "cm_W_extended.png")
    cm_full.to_csv(out_dir / "cm_W_extended.csv")
    print("\n[W — extended confusion matrix (incl. extra truth labels)]")
    print(cm_full)

    res["cm_extended"] = cm_full
    return res


# ---------------------------------------------------------------------------
# Step 3: FS classifier — only rows where W predicted T1W or T2W
# ---------------------------------------------------------------------------

def analyse_fs(df: pd.DataFrame, out_dir: Path) -> dict:
    truth_w = normalise_str(df["Clase W Final"])
    pred_w  = normalise_str(df["Predicción Clases W"])
    # Cascade gating: row must be W-eligible (truth in {T1W, T2W, DW}) AND
    # the W head must have predicted T1W or T2W (the FS head's actual input).
    mask = truth_w.isin(W_IN_CLASS) & pred_w.isin(FS_INPUT_W)
    sub = df.loc[mask]
    print(f"\nFS head input: {int(mask.sum()):,}/{len(df):,} rows "
          f"(truth W in {sorted(W_IN_CLASS)} AND pred W in {sorted(FS_INPUT_W)})")

    truth = fs_truth(sub)
    pred  = normalise_str(sub["Predicción Clases FS"])
    prob  = pd.to_numeric(sub["Predicción Clases FS P"], errors="coerce")
    probas = parse_probas(sub, PROBAS_COLS["FS"])

    return evaluate_head(
        name="FS",
        truth=truth, pred=pred, prob=prob,
        classes=["Y", "N"],
        out_dir=out_dir,
        proba_matrix=probas,
    )


# ---------------------------------------------------------------------------
# Step 4: C classifier — only rows where W predicted T1W
# ---------------------------------------------------------------------------

def analyse_c(df: pd.DataFrame, out_dir: Path) -> dict:
    truth_w = normalise_str(df["Clase W Final"])
    pred_w  = normalise_str(df["Predicción Clases W"])
    # Cascade gating: W-eligible truth AND the W head predicted T1W (the C
    # head's actual input).
    mask = truth_w.isin(W_IN_CLASS) & pred_w.isin(C_INPUT_W)
    sub = df.loc[mask]
    print(f"\nC head input: {int(mask.sum()):,}/{len(df):,} rows "
          f"(truth W in {sorted(W_IN_CLASS)} AND pred W in {sorted(C_INPUT_W)})")

    truth = normalise_str(sub["Clase C Final"])
    pred  = normalise_str(sub["Predicción Clases C"])
    prob  = pd.to_numeric(sub["Predicción Clases C P"], errors="coerce")
    probas = parse_probas(sub, PROBAS_COLS["C"])

    return evaluate_head(
        name="C",
        truth=truth, pred=pred, prob=prob,
        classes=["Y", "N"],
        out_dir=out_dir,
        proba_matrix=probas,
    )


# ---------------------------------------------------------------------------
# Step 5: Composite cascade label
# ---------------------------------------------------------------------------

def _cascade_label(w: str, fs: str, c: str) -> str:
    """
    Build the cascade-aware composite label.
      T1W -> T1W_{FS|nFS}_{CE|nCE}
      T2W -> T2W_{FS|nFS}        (C dimension collapsed)
      DW  -> DWI
      else -> raw W (caller filters)
    Anything other than fs == 'Y' counts as nFS (covers N and '-').
    Same for c.
    """
    if w == "T1W":
        return f"T1W_{'FS' if fs == 'Y' else 'nFS'}_{'CE' if c == 'Y' else 'nCE'}"
    if w == "T2W":
        return f"T2W_{'FS' if fs == 'Y' else 'nFS'}"
    if w in ("DW", "DWI"):
        return "DWI"
    return w  # T2*, PD, etc. — caller will filter


def _composite_series(w: pd.Series, fs: pd.Series, c: pd.Series) -> pd.Series:
    return pd.Series(
        [_cascade_label(*x) for x in zip(w, fs, c)],
        index=w.index,
    )


def analyse_composite(df: pd.DataFrame, out_dir: Path) -> dict:
    truth = _composite_series(
        normalise_str(df["Clase W Final"]),
        fs_truth(df),
        normalise_str(df["Clase C Final"]),
    )
    pred = _composite_series(
        normalise_str(df["Predicción Clases W"]),
        normalise_str(df["Predicción Clases FS"]),
        normalise_str(df["Predicción Clases C"]),
    )

    # Keep only rows whose TRUTH composite is one of the six final labels.
    # (Drops DWI, T2*, PD, etc. on the truth side.)
    keep = truth.isin(FINAL_LABELS)
    truth_k = truth[keep]
    pred_k  = pred[keep]
    print(f"\nComposite scope: {int(keep.sum()):,}/{len(df):,} rows "
          f"(truth composite in the six final labels)")

    # Order pred columns: target labels first, any off-target preds after
    pred_extra = [l for l in pred_k.unique() if l not in FINAL_LABELS]
    pred_order = FINAL_LABELS + sorted(pred_extra,
                                       key=lambda l: -int((pred_k == l).sum()))
    cm = confusion_table(truth_k, pred_k,
                         truth_order=FINAL_LABELS, pred_order=pred_order)
    metrics = per_class_metrics(cm).reindex(FINAL_LABELS)

    acc = overall_accuracy(truth_k, pred_k)

    # Composite confidence = product of head probabilities. For T2W rows,
    # the C head's reported probability is still in the file; multiplying
    # by it is harmless because it's a fixed scaling per row.
    prob_w  = pd.to_numeric(df.loc[keep, "Predicción Clases W P"],  errors="coerce")
    prob_fs = pd.to_numeric(df.loc[keep, "Predicción Clases FS P"], errors="coerce")
    prob_c  = pd.to_numeric(df.loc[keep, "Predicción Clases C P"],  errors="coerce")
    combined_prob = (prob_w * prob_fs * prob_c).clip(0, 1).values
    correct = (truth_k.values == pred_k.values).astype(int)
    comp_conf_auc = _safe_auc(correct, combined_prob)

    macro_prec = float(metrics["precision"].mean(skipna=True))
    macro_rec  = float(metrics["recall"].mean(skipna=True))
    macro_f1   = float(metrics["f1"].mean(skipna=True))

    print(f"\n[Composite (final cascade labels)]  n={len(truth_k):,}")
    print(f"  exact-match accuracy = {acc:.3f}")
    print(f"  macro-precision      = {macro_prec:.3f}")
    print(f"  macro-sensitivity    = {macro_rec:.3f}")
    print(f"  macro-F1             = {macro_f1:.3f}")
    print(f"  composite confidence AUC (p_W * p_FS * p_C) = {comp_conf_auc:.3f}")
    print(f"  confusion matrix:\n{cm}")
    print(f"  per-class metrics:\n{metrics.round(3)}")

    plot_confusion(cm, "Composite (cascade) — confusion matrix",
                   out_dir / "cm_composite.png")
    cm.to_csv(out_dir / "cm_composite.csv")
    metrics.to_csv(out_dir / "metrics_composite.csv")

    return dict(name="Composite", n=len(truth_k),
                accuracy=acc,
                macro_precision=macro_prec,
                macro_recall=macro_rec,
                macro_f1=macro_f1,
                confidence_auc=comp_conf_auc,
                cm=cm, metrics=metrics)


# ---------------------------------------------------------------------------
# Misclassified rows export
# ---------------------------------------------------------------------------

def save_errors(df: pd.DataFrame, out_dir: Path) -> None:
    key_cols = ["Paciente", "Estudio", "Serie", "Nombre DICOM"]
    truth_w  = normalise_str(df["Clase W Final"])
    truth_fs = fs_truth(df)
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

def _summary_row(res: dict) -> dict:
    return dict(stage=res["name"], n=res["n"],
                accuracy=res["accuracy"],
                macro_precision=res["macro_precision"],
                macro_sensitivity=res["macro_recall"],
                macro_f1=res["macro_f1"],
                macro_ovr_auc=res.get("macro_ovr_auc", float("nan")),
                confidence_auc=res["confidence_auc"])


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

    # Step 0 + 1: load, fix, combine, dedupe, sort
    df = combine(args.csv_a, args.csv_b)
    df.to_csv(args.out_dir / "combined_reviewed.csv", index=False)

    # Step 1 continued: drop non-target W truth buckets
    before = len(df)
    drop_mask = normalise_str(df["Clase W Final"]).isin(W_TRUTH_DROP)
    df = df.loc[~drop_mask].reset_index(drop=True)
    print(f"Dropped {int(drop_mask.sum()):,} rows with W truth in "
          f"{sorted(W_TRUTH_DROP)}  ({before:,} -> {len(df):,} remaining)\n")

    # Steps 2 - 5
    w_res    = analyse_w(df, args.out_dir)
    fs_res   = analyse_fs(df, args.out_dir)
    c_res    = analyse_c(df, args.out_dir)
    comp_res = analyse_composite(df, args.out_dir)

    # Summary table
    summary = pd.DataFrame([
        _summary_row(w_res),
        _summary_row(fs_res),
        _summary_row(c_res),
        _summary_row(comp_res),
    ])
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    print("\n========== Summary ==========")
    print(summary.round(3).to_string(index=False))
    print(f"\nSummary saved -> {args.out_dir / 'summary.csv'}")

    save_errors(df, args.out_dir)


if __name__ == "__main__":
    main()
