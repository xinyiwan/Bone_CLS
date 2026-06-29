"""
Inspect the synthetic-dot dataset BEFORE training.

Answers three questions:
  1. Is the dataset balanced and are dots actually injected into positives?
  2. Is the dot the brightest, localized thing -- i.e. is the label *reachable*?
  3. What does it look like? (saves mid-slice / max-projection PNGs)

If the simple "global max intensity" feature already separates the classes
(AUC near 1.0), a correct model must be able to learn the task. If even that
feature is at ~0.5, the synthetic signal itself is broken -- fix the data, not
the model.

Usage:
    python inspect_data.py --n 64 --out inspect_out
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from dataset import SyntheticDotDataset, AugConfig
from train import roc_auc          # reuse the same rank-based AUC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--img-size", type=int, nargs=3, default=[64, 64, 64])
    ap.add_argument("--radius", type=int, default=4)
    ap.add_argument("--dot-frac", type=float, default=0.95)
    ap.add_argument("--bg-ceiling", type=float, default=0.6)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--out", default="inspect_out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ds = SyntheticDotDataset(
        args.n, img_size=tuple(args.img_size), radius=args.radius,
        dot_frac=args.dot_frac, bg_ceiling=args.bg_ceiling,
        augment_cfg=AugConfig() if args.augment else None, seed=args.seed)

    labels, gmax, gmean, n_bright = [], [], [], []
    pos_vols, neg_vols = [], []
    for i in range(len(ds)):
        x, y = ds[i]
        v = x.squeeze(0).numpy()
        y = float(y)
        labels.append(y)
        gmax.append(float(v.max()))
        gmean.append(float(v.mean()))
        n_bright.append(int((v >= 0.9 * v.max()).sum()))
        if y > 0.5 and len(pos_vols) < 4:
            pos_vols.append(v)
        elif y <= 0.5 and len(neg_vols) < 4:
            neg_vols.append(v)

    labels = np.array(labels)
    gmax, gmean, n_bright = map(np.array, (gmax, gmean, n_bright))

    print(f"samples: {len(ds)}   positives: {int(labels.sum())}   "
          f"negatives: {int((labels == 0).sum())}")
    print("\n             mean(pos)     mean(neg)   separability-AUC")
    for name, feat in [("global max ", gmax),
                       ("global mean", gmean),
                       ("n_bright   ", n_bright.astype(float))]:
        auc = roc_auc(labels, feat)
        print(f"  {name}  {feat[labels==1].mean():10.4f}   "
              f"{feat[labels==0].mean():10.4f}   {auc:.3f}")

    print("\nInterpretation:")
    print("  * If any separability-AUC ~1.0 -> the dot is a clean, learnable")
    print("    signal; a failing model is a MODEL/training bug.")
    print("  * If all ~0.5 -> the synthetic signal is camouflaged; fix the")
    print("    data generator (dot must be the brightest, localized region).")

    # --- visual check: central slice + max projection for a few samples ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = [("POS", pos_vols), ("NEG", neg_vols)]
        n_show = min(4, min(len(pos_vols), len(neg_vols)))
        fig, axes = plt.subplots(2, n_show, figsize=(3 * n_show, 6))
        for r, (tag, vols) in enumerate(rows):
            for c in range(n_show):
                v = vols[c]
                mip = v.max(axis=0)                 # max-intensity projection over Z
                ax = axes[r, c] if n_show > 1 else axes[r]
                ax.imshow(mip, cmap="gray", vmin=0, vmax=1)
                ax.set_title(f"{tag} (max={v.max():.2f})")
                ax.axis("off")
        plt.tight_layout()
        path = os.path.join(args.out, "samples_mip.png")
        plt.savefig(path, dpi=120)
        print(f"\nsaved montage (Z max-projection) -> {path}")
        print("  -> a positive should show a clear bright blob; a negative none.")
    except Exception as e:                          # pragma: no cover
        print(f"\n(skipped montage: {e})")


if __name__ == "__main__":
    main()
