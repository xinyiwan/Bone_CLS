"""
QC: verify mask <-> image alignment for the segmentation set.

Two checks, from the *source* tree (the source of truth, before nnU-Net conversion):

  1. Geometry report (cheap, all pairs): for every (image, mask) pair, does the
     mask share the image's shape AND affine? A shape match with an AFFINE
     MISMATCH is the dangerous case -- `to_nnunet.py` overwrites the mask affine
     with the image's, so such a mask would be silently voxel-misaligned and
     would cap nnU-Net dice near zero.

  2. Visual overlay (sampled): mid-lesion slices along each axis with the mask
     overlaid in red, so you can eyeball whether the mask lands on the lesion.

Usage:
    python qc_overlay.py <root> --out-dir qc_out --n 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to

from pairs import find_pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("qc_out"))
    ap.add_argument("--n", type=int, default=20, help="cases to render overlays for")
    ap.add_argument("--all", action="store_true", help="render overlays for ALL pairs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pairs = [(s, sub, sc, ip, sp) for s, sub, sc, ip, sp, _src in find_pairs(args.root)
             if ip is not None]
    print(f"{len(pairs)} (image, mask) pairs\n")

    # --- 1. geometry report over ALL pairs (affine is cheap; no data load) ---
    n_shape_bad = n_affine_bad = 0
    bad_rows = []
    for subject, session, scan, image_path, seg_path in pairs:
        img = nib.load(str(image_path))
        seg = nib.load(str(seg_path))
        shape_ok = seg.shape[:3] == img.shape[:3]
        affine_ok = np.allclose(seg.affine, img.affine, atol=1e-3)
        if not shape_ok:
            n_shape_bad += 1
        elif not affine_ok:                  # same shape but different affine = danger
            n_affine_bad += 1
            bad_rows.append(f"{subject}/{scan}: shape ok, AFFINE differs")
    print(f"shape mismatches:                 {n_shape_bad}")
    print(f"shape-ok but AFFINE mismatches:   {n_affine_bad}  "
          f"(<- these get silently misaligned by to_nnunet.py)")
    for r in bad_rows[:20]:
        print("   ", r)
    if n_affine_bad:
        print("   -> resample masks into image space instead of copying the affine.")
    print()

    # --- 2. visual overlays (green segmentation contour) ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                   # pragma: no cover
        print(f"(skipping overlays: {e})")
        return

    if args.all:
        idx = range(len(pairs))
    else:
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(len(pairs))[: args.n]
    for k in idx:
        subject, session, scan, image_path, seg_path = pairs[k]
        img_nii = nib.load(str(image_path))
        img = img_nii.get_fdata(dtype=np.float32)
        # Resample the mask into the image grid (same operation to_nnunet.py uses),
        # so this overlay reflects the labels nnU-Net will actually train on.
        seg_nii = resample_from_to(nib.load(str(seg_path)),
                                   (img_nii.shape[:3], img_nii.affine), order=0)
        seg = (np.asanyarray(seg_nii.dataobj) > 0).astype(np.uint8)
        if seg.sum() == 0:
            continue
        com = np.argwhere(seg).mean(0).astype(int)          # lesion centre (i,j,k)

        fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
        for ax_i, axis in enumerate(range(3)):
            sl = com[axis]
            im_slice = np.take(img, sl, axis=axis)
            sg_slice = np.take(seg, sl, axis=axis)
            vmax = np.percentile(im_slice, 99) or 1.0
            axes[ax_i].imshow(im_slice.T, cmap="gray", vmin=0, vmax=vmax, origin="lower")
            if sg_slice.any():
                axes[ax_i].contour(sg_slice.T, levels=[0.5], colors="lime",
                                   linewidths=1.0)
            axes[ax_i].set_title(f"axis {axis} @ {sl}")
            axes[ax_i].axis("off")
        fig.suptitle(f"{subject}/{scan}", fontsize=9)
        fig.tight_layout()
        out = args.out_dir / f"{subject}_{scan}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
    scope = "ALL" if args.all else f"{args.n} sampled"
    print(f"overlays ({scope}, green contour) -> {args.out_dir}")


if __name__ == "__main__":
    main()
