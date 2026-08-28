"""
Score the shape probe's log-probabilities and hidden states -- CPU only.

Consumes `score_logprobs.py` output and answers three separate questions that
the generative accuracy number conflates:

  1. --logprobs   Is the model's failure a THRESHOLD problem?
                  Reports raw argmax accuracy, then re-thresholds the same
                  log-probs three ways and reports each. If accuracy jumps, the
                  signal was always there and the decision rule was wrong.

  2. --embeddings Is the signal in the REPRESENTATION at all?
                  Logistic regression on pooled hidden states, grouped by
                  case_id. This is the ceiling any amount of prompting or
                  head-only fine-tuning can reach. If the binary
                  irregular-vs-lobulated probe lands near chance, the
                  discriminating detail is not surviving the vision encoder and
                  the fix is input-side (crop tighter, upscale), not LoRA.

  3. --geometry   Is the GENERATOR's distinction even learnable?
                  Fits the same classifier on the shape_params column that
                  build_shapes.py recorded. This must come out near 1.0; if it
                  does not, the classes overlap by construction and no model
                  result about them means anything.

The threshold-free number to read first is the pairwise AUC. Accuracy depends on
where the threshold sits; AUC asks only whether the model RANKS true irregulars
above true lobulateds. AUC ~0.5 means calibration cannot help and neither can a
bias-only fine-tune. AUC ~0.8 with 0.02 recall means the entire deficit is the
threshold, and both calibration and LoRA will pay off immediately.

Runs in the ROOT uv environment (needs scikit-learn, which medgemma_pilot
deliberately does not depend on):

    uv run python working/vision_model/shape_probe/calibrate.py \
        --logprobs /results/shape_probe/clinical/logprobs_0shot.csv --geometry

    uv run python working/vision_model/shape_probe/calibrate.py \
        --embeddings /results/shape_probe/clinical/embeddings.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def report(name: str, y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> float:
    """Print accuracy, balanced accuracy, per-class precision/recall, prediction
    marginals and the confusion matrix. Returns balanced accuracy.

    Balanced accuracy and the prediction marginals are the headline numbers, not
    plain accuracy: a model that abandons one class and dumps it into a
    same-sized neighbour can gain plain accuracy while getting strictly worse at
    the task, which is exactly what the k=5 few-shot run did.
    """
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix

    acc = float((y_true == y_pred).mean())
    bal = float(balanced_accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))

    print(f"\n=== {name} ===")
    print(f"accuracy {acc:.3f}   balanced accuracy {bal:.3f}   "
          f"chance {1/len(labels):.3f}   n={len(y_true)}")
    print(f"{'class':<12}{'n':>5}{'recall':>9}{'precision':>11}{'predicted':>11}")
    for i, lab in enumerate(labels):
        n = int(cm[i].sum())
        npred = int(cm[:, i].sum())
        rec = cm[i, i] / n if n else float("nan")
        prec = cm[i, i] / npred if npred else float("nan")
        print(f"{lab:<12}{n:>5}{rec:>9.3f}{prec:>11.3f}{npred:>11}")
    print("confusion (rows = true, cols = predicted):")
    print(pd.DataFrame(cm, index=labels, columns=labels).to_string())
    return bal


def pairwise_auc(logp: np.ndarray, y: np.ndarray, labels: List[str],
                 groups: Optional[np.ndarray] = None,
                 difficulty: Optional[np.ndarray] = None) -> None:
    """AUC of the (a vs b) margin on the subset truly in {a, b}, for every pair.

    Threshold-free, so it is unaffected by the label-prior problem entirely: it
    measures only whether the model's score ORDERS the two classes correctly.
    This is the number that decides whether recalibration or fine-tuning can
    help, so it is reported before and independently of any calibration.

    Two things are reported alongside the point estimate because the headline
    number alone has misled us here before:

    `eff n` -- the number of distinct case_id groups, not images. Several
    difficulty levels are built from one source crop, so images are NOT
    independent and the naive CI on n images is too narrow. The rough CI printed
    uses the group count, which is the honest denominator.

    the difficulty split -- the sharp test of "saturated threshold" vs "cannot
    see it". `--difficulty` scales the deformation amplitude, so if the model
    perceives the cue at all, AUC must RISE with difficulty even when recall
    cannot move because the prior pins it. A flat AUC across difficulty while
    the amplitude doubles is strong evidence the cue is not being perceived.
    """
    from sklearn.metrics import roc_auc_score

    def one(mask: np.ndarray, i: int, j: int) -> Optional[tuple]:
        if mask.sum() < 4 or len(set(y[mask])) < 2:
            return None
        auc = roc_auc_score((y[mask] == j).astype(int), logp[mask, j] - logp[mask, i])
        n_eff = len(np.unique(groups[mask])) if groups is not None else int(mask.sum())
        # Hanley-McNeil SE, with the GROUP count as the per-class denominator.
        n1 = max(n_eff // 2, 2)
        q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
        se = float(np.sqrt(max(auc * (1 - auc) + (n1 - 1) * (q1 + q2 - 2 * auc ** 2), 0)
                           / (n1 * n1)))
        return auc, int(mask.sum()), n_eff, se

    print("\n=== pairwise ranking AUC (threshold-free; 0.5 = no signal) ===")
    print("    CI uses the case_id GROUP count, not the image count: several")
    print("    difficulty levels share a source crop, so images are not independent.")
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            r = one((y == i) | (y == j), i, j)
            if r is None:
                continue
            auc, n, n_eff, se = r
            lo, hi = max(auc - 1.96 * se, 0.0), min(auc + 1.96 * se, 1.0)
            flag = "" if lo > 0.5 else "   <- CI includes 0.5: no reliable signal"
            print(f"  {labels[i]:>11} vs {labels[j]:<11} n={n:>4} (eff {n_eff:>3})  "
                  f"AUC {auc:.3f}  95% CI [{lo:.3f}, {hi:.3f}]{flag}")

    if difficulty is None or len(np.unique(difficulty)) < 2:
        return
    levels = sorted(np.unique(difficulty), key=lambda v: float(v))
    print("\n=== the same AUC, split by difficulty (deformation amplitude) ===")
    print("    RISING with difficulty = the cue is perceived, the threshold is just wrong.")
    print("    FLAT across difficulty  = doubling the cue changes nothing; not perceived.")
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            cells = []
            for d in levels:
                r = one(((y == i) | (y == j)) & (difficulty == d), i, j)
                cells.append(f"d={d}: {r[0]:.3f}" if r else f"d={d}:   -- ")
            print(f"  {labels[i]:>11} vs {labels[j]:<11}  " + "   ".join(cells))


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------
def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def prior_correction(logp: np.ndarray) -> np.ndarray:
    """Divide out the model's content-independent label prior (Zhao et al. 2021).

    b_y = log(mean_x p_y(x)): the average mass the model puts on class y across
    the whole set. Subtracting it makes every class compete from an equal
    footing. Uses NO labels, so it is legitimate to apply to the full test set --
    it is a property of the model's output distribution, not of the answers.
    """
    p = softmax(logp)
    return logp - np.log(p.mean(axis=0, keepdims=True) + 1e-12)


def marginal_matching(logp: np.ndarray, target: Optional[np.ndarray] = None,
                      iters: int = 500, lr: float = 0.5) -> np.ndarray:
    """Fit a per-class bias so the predicted marginal matches `target`.

    Uses only the DESIGN of the experiment, not the labels: build_shapes.py emits
    a balanced set (one random class per source row, per level), so a uniform
    predicted marginal is the right target. Still label-free, but a stronger
    correction than prior_correction because it matches the marginal exactly
    instead of approximating it with one multiplicative step.

    Fixed-point iteration on b_y += lr * log(current_y / target_y), damped by
    `lr`. Operating on soft probabilities rather than hard argmax counts keeps
    the update continuous and convergent.
    """
    K = logp.shape[1]
    target = np.full(K, 1.0 / K) if target is None else target / target.sum()
    b = np.zeros(K)
    for _ in range(iters):
        cur = softmax(logp - b).mean(axis=0)
        b += lr * (np.log(cur + 1e-12) - np.log(target + 1e-12))
    return logp - b


def matrix_scaling(logp: np.ndarray, y: np.ndarray, groups: np.ndarray,
                   n_splits: int = 5) -> np.ndarray:
    """Learned rescaling: multinomial logistic regression on the K log-probs.

    A K x K matrix plus K biases (12 parameters for 3 classes) -- the standard
    "matrix scaling" calibration family, which contains temperature scaling and
    per-class bias as special cases. Fitted out-of-fold with GroupKFold on
    case_id so a source crop never appears in both the fit and the evaluation;
    without that grouping the several difficulty levels built from one crop leak
    and the reported gain is inflated.

    Returns out-of-fold decision scores, so the printed accuracy is honest.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    n_splits = min(n_splits, len(np.unique(groups)))
    if n_splits < 2:
        raise SystemExit("need >=2 distinct case_id groups for grouped CV")
    oof = np.zeros_like(logp)
    for tr, te in GroupKFold(n_splits=n_splits).split(logp, y, groups):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(logp[tr], y[tr])
        d = clf.decision_function(logp[te])
        # Binary LogisticRegression returns a 1-D margin; widen it back to K cols.
        oof[te] = d if d.ndim == 2 else np.column_stack([-d, d])
    return oof


# --------------------------------------------------------------------------
# linear probe
# --------------------------------------------------------------------------
def probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray, labels: List[str],
          name: str, n_splits: int = 5) -> float:
    """Out-of-fold logistic-regression probe on frozen features.

    Standardised then L2-regularised, grouped by case_id. The point is not to
    build a classifier -- it is to ask whether a LINEAR readout of the frozen
    representation separates the classes. If it does and the VLM's own answers do
    not, the deficit is in the readout, which is what LoRA adjusts. If it does
    not, the information is absent from the representation and LoRA has nothing
    to recalibrate.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n_splits = min(n_splits, len(np.unique(groups)))
    if n_splits < 2 or len(np.unique(y)) < 2:
        print(f"\n=== {name} === skipped (need >=2 groups and >=2 classes)")
        return float("nan")
    pred = np.zeros(len(y), dtype=int)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=5000, C=0.1))
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
    return report(name, y, pred, labels)


def parse_shape_params(series: pd.Series, y: np.ndarray) -> pd.DataFrame:
    """'lobe_k=5;lobe_amp=0.24' -> a numeric feature frame, class-indicator
    columns REMOVED.

    Classes come from different branches of the radial equation, so they do not
    share a parameter set: `lobe_k` is only ever written for lobulated,
    `jag_amp` only for irregular. Keeping those and filling the gaps with 0 makes
    the sanity check vacuous -- a classifier separates the classes perfectly from
    which columns are merely PRESENT, without looking at a single value, and
    reports 1.000 no matter how much the geometry actually overlaps.

    So drop any column that is absent for an entire class. What survives is the
    parameters defined for every class, which is the only set on which "are these
    classes separable by their geometry" is a real question. If nothing survives,
    say so rather than printing a meaningless 1.000: the honest version of this
    check then needs descriptors recomputed from r(theta) (dom_k, kurt,
    peak-to-peak -- the README's table), which are defined for all three classes.
    """
    rows: List[Dict[str, float]] = []
    present: List[set] = []
    for val in series.fillna("").astype(str):
        d: Dict[str, float] = {}
        for part in val.split(";"):
            if "=" not in part:
                continue
            k, _, v = part.partition("=")
            try:
                d[k.strip()] = float(v)
            except ValueError:
                pass
        rows.append(d)
        present.append(set(d))

    df = pd.DataFrame(rows)
    shared = [c for c in df.columns
              if all(any(c in present[i] for i in np.where(y == k)[0]) for k in np.unique(y))]
    dropped = [c for c in df.columns if c not in shared]
    if dropped:
        print(f"dropping class-indicator parameter(s) absent for some class: {dropped}")
    return df[shared].fillna(0.0)


# --------------------------------------------------------------------------
# drivers
# --------------------------------------------------------------------------
def run_logprobs(paths: List[Path], do_geometry: bool) -> None:
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    cols = [c for c in df.columns if c.startswith("logp_")]
    if not cols:
        raise SystemExit(f"no logp_* columns in {paths}; is this a score_logprobs.py --mode score CSV?")
    labels = [c[len("logp_"):] for c in cols]
    logp = df[cols].to_numpy(dtype=np.float64)
    y = np.array([labels.index(s) for s in df["shape"].astype(str)])
    groups = df["case_id"].astype(str).to_numpy()

    print(f"{len(df)} image(s), {len(labels)} classes {labels}, "
          f"{len(np.unique(groups))} distinct case_id group(s)")
    if "scored_after_thinking" in df.columns and df["scored_after_thinking"].astype(str).eq("1").any():
        print("scored AFTER a replayed thinking block (same path as --mode infer)")

    # AUC first: it is the only number here that a bad threshold cannot spoil.
    diff = df["difficulty"].astype(str).to_numpy() if "difficulty" in df.columns else None
    pairwise_auc(logp, y, labels, groups=groups, difficulty=diff)

    report("raw argmax (no calibration)", y, logp.argmax(1), labels)
    report("prior correction (label-free)", y, prior_correction(logp).argmax(1), labels)
    report("marginal matching to uniform (label-free, uses the balanced design)",
           y, marginal_matching(logp).argmax(1), labels)
    report("matrix scaling (12 params, out-of-fold, grouped by case_id)",
           y, matrix_scaling(logp, y, groups).argmax(1), labels)

    if "difficulty" in df.columns and df["difficulty"].nunique() > 1:
        print("\n=== balanced accuracy by difficulty, before vs after marginal matching ===")
        cal = marginal_matching(logp)
        from sklearn.metrics import balanced_accuracy_score
        for d, idx in df.groupby(df["difficulty"].astype(str)).groups.items():
            m = df.index.isin(idx)
            print(f"  difficulty {d:>6}  n={int(m.sum()):>4}  "
                  f"raw {balanced_accuracy_score(y[m], logp[m].argmax(1)):.3f}  ->  "
                  f"calibrated {balanced_accuracy_score(y[m], cal[m].argmax(1)):.3f}")

    if do_geometry:
        print("\n############ generator-parameter sanity check ############")
        G = parse_shape_params(df["shape_params"], y)
        if G.empty or G.shape[1] == 0:
            print("no parameter is recorded for every class, so this check cannot run on "
                  "shape_params alone. Recompute the shared descriptors from r(theta) "
                  "(dom_k / kurt / peak-to-peak) and re-run to get a meaningful number.")
        else:
            print(f"shared generator features: {list(G.columns)}")
            probe(G.to_numpy(dtype=np.float64), y, groups, labels,
                  "generator-parameter separability (expect ~1.0; if lower, the classes "
                  "overlap by construction)")


def run_embeddings(paths: List[Path]) -> None:
    metas: List[pd.DataFrame] = []
    arrays: Dict[str, List[np.ndarray]] = {}
    labels: List[str] = []
    for p in paths:
        z = np.load(p, allow_pickle=False)
        found = [str(s) for s in z["labels"]]
        if labels and found != labels:
            raise SystemExit(f"{p} has labels {found}, expected {labels}; do not mix builds")
        labels = found
        metas.append(pd.read_csv(p.with_suffix(".meta.csv")))
        for key in z.files:
            if key != "labels":
                arrays.setdefault(key, []).append(z[key])
    meta = pd.concat(metas, ignore_index=True)
    y = np.array([labels.index(s) for s in meta["shape"].astype(str)])
    groups = meta["case_id"].astype(str).to_numpy()

    print(f"{len(meta)} image(s), {len(np.unique(groups))} case_id group(s), "
          f"features: {sorted(arrays)}")

    for key in sorted(arrays):
        X = np.concatenate(arrays[key])
        print(f"\n############ features: {key}  (dim {X.shape[1]}) ############")
        probe(X, y, groups, labels, f"{key}: all {len(labels)} classes")
        # The binary that carries essentially all of the error mass. Reported
        # separately because a 3-class probe can look mediocre for the wrong
        # reason -- an easy round_oval class propping up a collapsed pair.
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                m = (y == i) | (y == j)
                probe(X[m], (y[m] == j).astype(int), groups[m],
                      [labels[i], labels[j]], f"{key}: {labels[i]} vs {labels[j]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logprobs", type=Path, nargs="*", default=[],
                    help="score_logprobs.py --mode score CSV(s); pass every shard")
    ap.add_argument("--embeddings", type=Path, nargs="*", default=[],
                    help="score_logprobs.py --mode embed NPZ(s); each needs its .meta.csv beside it")
    ap.add_argument("--geometry", action="store_true",
                    help="also fit the sanity-check classifier on the shape_params column")
    args = ap.parse_args()

    if not args.logprobs and not args.embeddings:
        ap.error("pass --logprobs and/or --embeddings")
    if args.logprobs:
        run_logprobs(args.logprobs, args.geometry)
    if args.embeddings:
        run_embeddings(args.embeddings)


if __name__ == "__main__":
    main()
