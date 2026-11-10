"""Compare BTI-RADS predictions with the report-based diagnosis (palabra_manual).

Diagnosis -> group uses working/data_ov/lesion_category.py (benign / intermediate /
malignant / mimicker). 'uncertain' and 'none' are dropped. Binary metrics:
four binary tasks (see TASKS); main = malignant+intermediate vs benign+mimicker. metrics.csv has\nsens/spec/precision/NPV/F1/accuracy at score>=4, >=5, prob>=50% and the Youden cut-off.
Outputs in --out: comparison.csv (one row per study), per_diagnosis.csv,
score_by_group.csv, confusion_matrix.png (group x score), auc/summary printed, prob_by_group.png, prob_by_diagnosis.png.

  python compare_bti.py
"""
import argparse, json, sys
import numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

R = Path("/Users/xinyi/Documents/github/Bone_CLS")
sys.path.insert(0, str(R / "working/data_ov"))
from labels import to_label
from lesion_category import to_category

GROUPS = ["benign", "mimicker", "intermediate", "malignant"]

ap = argparse.ArgumentParser()
ap.add_argument("--version", help="prediction version folder name, e.g. v2 (default: latest)")
ap.add_argument("--restrict-to", metavar="VERSION", help="only studies that also exist in this version "
                "(fair comparison across versions); outputs go to <version>_on_<VERSION>/")
ap.add_argument("--merged", default=R / "output/merged_data/kira-1104-combined_dataset.filtered.csv")
ap.add_argument("--subset", default="subset_pct15", help="analysis/<subset>/<version>/ holds the outputs")
a = ap.parse_args()
ver = a.version or sorted(p.name for p in (R / "output/BTI/prediction").glob("v*"))[-1]
a.pred = R / "output/BTI/prediction" / ver
keep = {f.parent.name for f in (R / "output/BTI/prediction" / a.restrict_to).glob("*/result.json")} if a.restrict_to else None
out = R / "output/BTI/analysis" / a.subset / (ver + (f"_on_{a.restrict_to}" if keep else "")); out.mkdir(parents=True, exist_ok=True)
print("version:", ver)

rows = []
for f in Path(a.pred).glob("*/result.json"):
    if keep is not None and f.parent.name not in keep:
        continue
    j = json.loads(f.read_text())
    rows.append({"case_id": f.parent.name, "bti_score": j["score"], "bti_prob": j["probability"],
                 "Age": j["request"]["Age"], "Size (mm)": j["request"]["Size (mm)"]})
pred = pd.DataFrame(rows)
m = pd.read_csv(a.merged, dtype=str)[["info_key", "subject_code", "palabra_manual"]]
df = pred.merge(m, left_on="case_id", right_on="info_key").drop(columns="info_key")
df["diagnosis"] = df.palabra_manual.map(to_label)
df["group"] = df.diagnosis.map(to_category)
df.sort_values(["group", "diagnosis", "case_id"]).to_csv(out / "prediction_gt.csv", index=False)
d = df[df.group.isin(GROUPS)]
print(f"{len(pred)} predictions, {len(d)} with a usable diagnosis (dropped {len(df) - len(d)} uncertain/none)")

d.groupby("diagnosis").agg(group=("group", "first"), n=("case_id", "size"), prob_median=("bti_prob", "median"),
                           prob_min=("bti_prob", "min"), prob_max=("bti_prob", "max"),
                           score_median=("bti_score", "median")).sort_values(["group", "prob_median"]).to_csv(out / "per_diagnosis.csv")
ct = pd.crosstab(d.group, d.bti_score).reindex(GROUPS).dropna(how="all").fillna(0).astype(int)
ct.to_csv(out / "score_by_group.csv")
fig, ax = plt.subplots(figsize=(4.5, 3.6))  # confusion matrix: diagnosis group x BTI score, shade = row %
pct = ct.div(ct.sum(axis=1), axis=0)
ax.imshow(pct.values, cmap="Blues", vmin=0, vmax=1)
for i in range(ct.shape[0]):
    for j in range(ct.shape[1]):
        ax.text(j, i, f"{ct.iat[i, j]}\n{pct.iat[i, j]:.0%}", ha="center", va="center",
                color="white" if pct.iat[i, j] > .6 else "black")
ax.set_xticks(range(ct.shape[1]), [f"BTI {int(c)}" for c in ct.columns])
ax.set_yticks(range(ct.shape[0]), [f"{g} (n={ct.loc[g].sum()})" for g in ct.index])
ax.set_xlabel("BTI-RADS score"); ax.set_ylabel("Diagnosis group (reports)")
fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=150, bbox_inches="tight")
print("\nBTI score by group:\n", pd.crosstab(d.group, d.bti_score).to_string())
print("\nmedian probability by group:\n", d.groupby("group").bti_prob.median().round(1).to_string())

def auc_ci(y, score, n=2000):  # bootstrap 95% CI, resampling studies (fixed seed: same CI every call)
    rng = np.random.default_rng(0)
    v = [roc_auc_score(y.iloc[k], score.iloc[k]) for k in (rng.integers(0, len(y), len(y)) for _ in range(n))
         if y.iloc[k].nunique() == 2]
    return np.percentile(v, [2.5, 97.5])


def cut_metrics(y, pred):
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    div = lambda a, b: a / b if b else float("nan")
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, sensitivity=div(tp, tp + fn), specificity=div(tn, tn + fp),
                precision=div(tp, tp + fp), NPV=div(tn, tn + fn), F1=div(2 * tp, 2 * tp + fp + fn),
                accuracy=(tp + tn) / len(y))


# binary tasks (positive vs negative groups). Main: malignant + intermediate (similar management) vs benign + mimicker
TASKS = [("MAIN malignant+intermediate vs benign+mimicker", ["malignant", "intermediate"], ["benign", "mimicker"],
          "roc_main_malig_interm_vs_rest.png"),
         ("malignant+intermediate vs benign (no mimickers)", ["malignant", "intermediate"], ["benign"],
          "roc_malig_interm_vs_benign.png"),
         ("malignant vs all others", ["malignant"], ["benign", "mimicker", "intermediate"], "roc_malignant_vs_rest.png"),
         ("malignant vs benign", ["malignant"], ["benign"], "roc_benign_vs_malignant.png")]
allm = []
pd.set_option("display.width", 200)
for task, pos_g, neg_g, fname in TASKS:
    b = d[d.group.isin(pos_g + neg_g)]
    y = (b.group.isin(pos_g)).astype(int)
    title = f"{'+'.join(pos_g)} (n={y.sum()}) vs {'+'.join(neg_g)} (n={(y == 0).sum()})"
    print(f"\n== {task}: {title}")
    aucs, auc_cols = {}, {}
    for name, col in [("BTI score", "bti_score"), ("BTI probability", "bti_prob")]:
        lo, hi = auc_ci(y, b[col]); aucs[col] = roc_auc_score(y, b[col])
        auc_cols[f"AUC_{col[4:]}"] = round(aucs[col], 3); auc_cols[f"AUC_{col[4:]}_CI"] = f"{lo:.2f}-{hi:.2f}"
        print(f"AUC {name}: {aucs[col]:.3f}  (95% CI {lo:.2f}-{hi:.2f})")
    fpr, tpr, thr = roc_curve(y, b.bti_prob)
    youden = thr[np.argmax(tpr - fpr)]  # data-derived, so optimistic
    rows = []
    for name, pred in [("score >= 4", b.bti_score >= 4), ("score >= 5", b.bti_score >= 5),
                       ("prob >= 50%", b.bti_prob >= 50), (f"prob >= {youden:.1f}% (Youden, data-derived)", b.bti_prob >= youden)]:
        rows.append({"task": task, "n_pos": int(y.sum()), "n_neg": int((y == 0).sum()), **auc_cols,
                     "cutoff": name, **cut_metrics(y, pred)})
    mt = pd.DataFrame(rows); allm.append(mt)
    print(mt.drop(columns=["task", "n_pos", "n_neg"]).round(2).to_string(index=False))
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, col, st in [("BTI probability", "bti_prob", "-"), ("BTI score", "bti_score", "--")]:
        f_, t_, _ = roc_curve(y, b[col]); lo, hi = auc_ci(y, b[col])
        ax.plot(f_, t_, st, lw=2, label=f"{name}: AUC {aucs[col]:.2f} ({lo:.2f}-{hi:.2f})")
    ax.plot([0, 1], [0, 1], ":", color="gray")
    ax.set(xlabel="1 - specificity", ylabel="Sensitivity", xlim=(0, 1), ylim=(0, 1.02),
           title=f"{'+'.join(pos_g)} (n={y.sum()})\nvs {'+'.join(neg_g)} (n={(y == 0).sum()})")
    ax.title.set_fontsize(9); ax.legend(loc="lower right", fontsize=8); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(out / fname, dpi=150)
pd.concat(allm).to_csv(out / "metrics.csv", index=False)

colors = {"benign": "#4c78a8", "mimicker": "#7f8c8d", "intermediate": "#f2a93b", "malignant": "#c0392b"}
fig, ax = plt.subplots(figsize=(6, 4))
for i, g in enumerate(GROUPS):
    v = d[d.group == g].bti_prob
    ax.boxplot(v, positions=[i], widths=0.5, showfliers=False)
    ax.scatter([i + (j % 7 - 3) * 0.04 for j in range(len(v))], v, s=14, color=colors[g], alpha=.8, zorder=3)
ax.set_xticks(range(4), GROUPS); ax.set_ylabel("BTI-RADS P(malignancy) %")
fig.tight_layout(); fig.savefig(out / "prob_by_group.png", dpi=150)

order = d.groupby("diagnosis").bti_prob.median().sort_values().index
fig, ax = plt.subplots(figsize=(7, 7))
for i, dx in enumerate(order):
    s = d[d.diagnosis == dx]
    ax.scatter(s.bti_prob, [i] * len(s), color=colors[s.group.iloc[0]], s=22, alpha=.85)
ax.set_yticks(range(len(order)), order); ax.set_xlabel("BTI-RADS P(malignancy) %")
fig.tight_layout(); fig.savefig(out / "prob_by_diagnosis.png", dpi=150)
print("\nsaved ->", out)
