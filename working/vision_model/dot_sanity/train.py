"""
Train ViT3D on the synthetic-dot sanity task and report PASS/FAIL.

The point of this script is *implementation verification*, not science: if the
ViT + data loader + training loop are wired correctly, the model should drive
validation AUC to ~1.0 on the trivial "is there a bright dot?" task within a
handful of epochs. If it cannot, the bug is in the plumbing, not the data.

Examples
--------
# Pure synthetic (no files needed) -- the default sanity check:
python train.py --mode synthetic --epochs 15 --n-train 256 --n-val 64

# Real volumes with injected dots (provide a CSV of img[,seg] paths):
python train.py --mode nifti --csv paths.csv --img-root /data/imgs --epochs 15
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import (AugConfig, NiftiDotDataset, SyntheticDotDataset,
                     split_indices)
from vit3d import ViT3D


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUC (no sklearn dependency)."""
    y_true = y_true.astype(int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    sum_pos = ranks[y_true == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def build_datasets(args):
    aug = AugConfig() if args.augment else None
    if args.mode == "synthetic":
        train = SyntheticDotDataset(
            args.n_train, img_size=tuple(args.img_size), radius=args.radius,
            dot_frac=args.dot_frac, bg_ceiling=args.bg_ceiling,
            augment_cfg=aug, seed=args.seed)
        val = SyntheticDotDataset(
            args.n_val, img_size=tuple(args.img_size), radius=args.radius,
            dot_frac=args.dot_frac, bg_ceiling=args.bg_ceiling,
            augment_cfg=None, seed=args.seed + 999)
        return train, val

    # nifti mode
    items = []
    with open(args.csv, newline="") as f:
        for row in csv.DictReader(f):
            img = row["img"]
            seg = row.get("seg") or None
            if args.img_root:
                img = os.path.join(args.img_root, img)
                if seg:
                    seg = os.path.join(args.img_root, seg)
            items.append({"img": img, "seg": seg})
    full = NiftiDotDataset(
        items, img_size=tuple(args.img_size), radius=args.radius,
        dot_frac=args.dot_frac, augment_cfg=aug, seed=args.seed)
    tr_idx, va_idx = split_indices(len(full), args.val_frac, args.seed)
    return Subset(full, tr_idx), Subset(full, va_idx)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x).squeeze(1)
        ps.append(torch.sigmoid(logits).cpu().numpy())
        ys.append(y.numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    acc = float(((p > 0.5).astype(float) == y).mean())
    return acc, roc_auc(y, p)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", choices=["synthetic", "nifti"], default="synthetic")
    ap.add_argument("--csv", help="nifti mode: CSV with columns img[,seg]")
    ap.add_argument("--img-root", default="", help="prefix for CSV paths")
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--n-train", type=int, default=256)
    ap.add_argument("--n-val", type=int, default=64)
    ap.add_argument("--img-size", type=int, nargs=3, default=[64, 64, 64])
    ap.add_argument("--patch-size", type=int, nargs=3, default=[16, 16, 16])
    ap.add_argument("--radius", type=int, default=4)
    ap.add_argument("--dot-frac", type=float, default=0.95)
    ap.add_argument("--bg-ceiling", type=float, default=0.6,
                    help="synthetic: background capped here; raise toward "
                         "--dot-frac to make the task harder")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pass-auc", type=float, default=0.95,
                    help="val AUC threshold for the sanity check to PASS")
    args = ap.parse_args()

    if args.mode == "nifti" and not args.csv:
        ap.error("--csv is required in nifti mode")

    set_seed(args.seed)
    device = pick_device(args.device)
    print(f"device: {device}")

    train_ds, val_ds = build_datasets(args)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    print(f"train: {len(train_ds)}  val: {len(val_ds)}")

    model = ViT3D(img_size=tuple(args.img_size), patch_size=tuple(args.patch_size),
                  in_chans=1, num_classes=1, dim=args.dim, depth=args.depth,
                  heads=args.heads).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.2f}M params, "
          f"{model.patch_embed.num_patches} patches")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    best_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x).squeeze(1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
        train_loss = running / len(train_ds)
        acc, auc = evaluate(model, val_loader, device)
        best_auc = max(best_auc, auc)
        print(f"epoch {epoch:3d} | train_loss {train_loss:.4f} | "
              f"val_acc {acc:.3f} | val_auc {auc:.3f}")

    passed = best_auc >= args.pass_auc
    print("\n" + "=" * 48)
    print(f"best val AUC = {best_auc:.3f}  (threshold {args.pass_auc})")
    print(f"SANITY CHECK: {'PASS ✅' if passed else 'FAIL ❌'}")
    print("=" * 48)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
