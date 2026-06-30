"""
Per-modality / per-orientation evaluation of nnU-Net validation predictions,
plus qualitative GT-vs-prediction slice overlays.

Reads each fold's ``validation/summary.json`` (per-case Dice and the predicted /
reference voxel counts) and joins it to the sequence + plane recorded in
``case_metadata.csv`` (written by to_nnunet.py). This answers:

  * Does Dice differ by sequence / by plane?               (issue 1)
  * Where do we oversegment? -> n_pred / n_ref ratio > 1   (issue 1)
  * Do GT and prediction actually look right?  -> overlays (issue 2)

Outputs (to --out-dir):
  per_case.csv            case, fold, sequence, plane, dice, n_pred, n_ref, ratio
  dice_by_sequence.csv    count / mean / median / std Dice + mean overseg ratio
  dice_by_plane.csv       same, by acquisition plane
  plots/                  dice boxplots by sequence & plane
  overlays/<sequence>/    one slice per case: image + GT (green) + pred (red)

nnU-Net folds are 0-indexed; "fold 1 & 2" usually means --folds 0 1.

Usage:
    python eval_folds.py \
        --results-dir $nnUNet_results/Dataset501_BoneTumour/nnUNetTrainer__nnUNetPlans__3d_fullres \
        --raw-dir     $nnUNet_raw/Dataset501_BoneTumour \
        --folds 0 1 --out-dir eval_out
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib


def load_fold_metrics(results_dir: Path, fold: int) -> list[dict]:
    """Parse validation/summary.json for one fold -> list of per-case rows."""
    summ = results_dir / f"fold_{fold}" / "validation" / "summary.json"
    if not summ.exists():
        raise SystemExit(f"missing {summ} — has fold {fold} finished validating?")
    data = json.loads(summ.read_text())
    rows = []
    for entry in data.get("metric_per_case", []):
        case = Path(entry["prediction_file"]).name
        for suf in (".nii.gz", ".nii"):
            if case.endswith(suf):
                case = case[: -len(suf)]
        m = entry["metrics"].get("1", {})               # foreground label "1"
        n_pred = float(m.get("n_pred", np.nan))
        n_ref = float(m.get("n_ref", np.nan))
        rows.append(dict(
            case=case, fold=fold,
            dice=float(m.get("Dice", np.nan)),
            n_pred=n_pred, n_ref=n_ref,
            ratio=(n_pred / n_ref if n_ref else np.nan),  # >1 => oversegment
        ))
    return rows


def summarise(df: pd.DataFrame, by: str) -> pd.DataFrame:
    g = df.groupby(by)
    out = pd.DataFrame({
        "n": g.size(),
        "dice_mean": g["dice"].mean(),
        "dice_median": g["dice"].median(),
        "dice_std": g["dice"].std(),
        "overseg_ratio_mean": g["ratio"].mean(),
    }).sort_values("dice_mean")
    return out.round(3)


def slice_axis_of(img: nib.Nifti1Image) -> int:
    """Acquisition slice axis = the one with the largest spacing."""
    return int(np.argmax(img.header.get_zooms()[:3]))


def make_overlays(df: pd.DataFrame, results_dir: Path, raw_dir: Path,
                  out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                              # pragma: no cover
        print(f"(skipping overlays: {e})")
        return

    for _, r in df.iterrows():
        case, fold = r["case"], int(r["fold"])
        img_p = raw_dir / "imagesTr" / f"{case}_0000.nii.gz"
        gt_p = raw_dir / "labelsTr" / f"{case}.nii.gz"
        pr_p = results_dir / f"fold_{fold}" / "validation" / f"{case}.nii.gz"
        if not (img_p.exists() and gt_p.exists() and pr_p.exists()):
            continue

        img_nii = nib.load(str(img_p))
        img = img_nii.get_fdata(dtype=np.float32)
        gt = np.asanyarray(nib.load(str(gt_p)).dataobj) > 0
        pr = np.asanyarray(nib.load(str(pr_p)).dataobj) > 0

        ax = slice_axis_of(img_nii)
        union = gt | pr
        if union.sum() == 0:
            continue
        # slice (along acquisition axis) with the largest GT|pred area
        areas = union.sum(axis=tuple(a for a in range(3) if a != ax))
        idx = int(np.argmax(areas))
        im = np.take(img, idx, axis=ax).T
        g = np.take(gt, idx, axis=ax).T
        p = np.take(pr, idx, axis=ax).T

        fig, axp = plt.subplots(figsize=(4.2, 4.2))
        vmax = np.percentile(im, 99) or 1.0
        axp.imshow(im, cmap="gray", vmin=0, vmax=vmax, origin="lower")
        if g.any():
            axp.contour(g, levels=[0.5], colors="lime", linewidths=1.2)
        if p.any():
            axp.contour(p, levels=[0.5], colors="red", linewidths=1.2)
        axp.set_title(f"{case}\n{r['sequence']} | {r['plane']} | "
                      f"Dice={r['dice']:.2f} | ratio={r['ratio']:.2f}", fontsize=8)
        axp.axis("off")
        # legend proxies
        from matplotlib.lines import Line2D
        axp.legend(handles=[Line2D([0], [0], color="lime", label="GT"),
                            Line2D([0], [0], color="red", label="pred")],
                   loc="lower right", fontsize=7, framealpha=0.5)
        fig.tight_layout()
        seq_dir = out_dir / "overlays" / str(r["sequence"]).replace("/", "-")
        seq_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(seq_dir / f"{case}.png", dpi=110)
        plt.close(fig)
    print(f"overlays -> {out_dir / 'overlays'}  (green=GT, red=pred)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results-dir", type=Path, required=True,
                    help=".../nnUNetTrainer__nnUNetPlans__3d_fullres")
    ap.add_argument("--raw-dir", type=Path, required=True,
                    help="$nnUNet_raw/Dataset501_BoneTumour")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--out-dir", type=Path, default=Path("eval_out"))
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(args.raw_dir / "case_metadata.csv")[
        ["case", "sequence", "plane"]]

    rows = []
    for f in args.folds:
        rows.extend(load_fold_metrics(args.results_dir, f))
    df = pd.DataFrame(rows).merge(meta, on="case", how="left")
    df.to_csv(args.out_dir / "per_case.csv", index=False)
    print(f"\n{len(df)} validation cases across folds {args.folds}")
    print(f"overall Dice: mean={df['dice'].mean():.3f}  median={df['dice'].median():.3f}\n")

    by_seq = summarise(df, "sequence")
    by_plane = summarise(df, "plane")
    by_seq.to_csv(args.out_dir / "dice_by_sequence.csv")
    by_plane.to_csv(args.out_dir / "dice_by_plane.csv")
    print("=== Dice by sequence ===");  print(by_seq, "\n")
    print("=== Dice by plane ===");     print(by_plane, "\n")
    print("(overseg_ratio_mean > 1 => predictions larger than GT on average)\n")

    if not args.no_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plots = args.out_dir / "plots"
            plots.mkdir(exist_ok=True)
            for col in ("sequence", "plane"):
                order = df.groupby(col)["dice"].median().sort_values().index
                data = [df.loc[df[col] == c, "dice"].values for c in order]
                fig, axb = plt.subplots(figsize=(max(5, len(order) * 1.1), 4))
                axb.boxplot(data, labels=list(order), showmeans=True)
                axb.set_ylabel("Dice")
                axb.set_title(f"Dice by {col} (folds {args.folds})")
                axb.tick_params(axis="x", rotation=30)
                axb.set_ylim(0, 1)
                fig.tight_layout()
                fig.savefig(plots / f"dice_by_{col}.png", dpi=120)
                plt.close(fig)
            print(f"boxplots -> {plots}")
        except Exception as e:                          # pragma: no cover
            print(f"(skipping boxplots: {e})")
        make_overlays(df, args.results_dir, args.raw_dir, args.out_dir)


if __name__ == "__main__":
    main()
