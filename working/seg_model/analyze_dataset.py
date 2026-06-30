"""
Dataset distribution analysis for the bone-tumour segmentation set.

Walks the dcm2nifti-style tree

    <root>/<subject>/<session>/<scan>/images.nii.gz
                              /segmentation_history/segs/<scan>_seg.nii.gz

i.e. the manual masks live in a per-session ``segmentation_history/segs/``
folder (one ``<scan>_seg.nii.gz`` per segmented sequence), NOT beside the image.
The ``FINAL_*`` / ``point_*`` / ``lasso_*`` files in segmentation_history are
editing history and are ignored.

Discovery is **segmentation-driven**: the masks in ``segs/`` define the labelled
set (some images are excluded during segmentation), and each mask is traced back
to its image. For every (mask, image) pair it records the
geometry and intensity facts that drive segmentation preprocessing:

  * acquisition plane   (sagittal / coronal / axial) -- derived from the affine
  * sequence type       (T1W / T2W / ... ) -- from a ground-truth table if given,
                        else parsed from the scan-folder name
  * shape per axis      (nx, ny, nz)
  * spacing per axis    (mm) + slice thickness (largest-spacing axis)
  * intensity stats     whole-image and within the segmentation foreground
  * label content       #foreground voxels, foreground fraction, #labels,
                        physical tumour volume (mm^3)

Outputs (to --out-dir):
  * per_scan.csv        one row per labelled scan (the analysis table)
  * summary.txt         aggregate stats + the preprocessing-relevant numbers
  * plots/*.png         histograms (size, spacing, intensity, fg-fraction) and
                        bar charts (plane, sequence)

These numbers tell you, before training:
  - whether scans share an orientation/spacing (-> 3d_fullres vs 2d, target
    spacing for resampling),
  - how skewed tumour size is (-> patch size, oversampling),
  - how different the sequences look (-> single-sequence vs pooled training),
  - the intensity range (-> normalisation scheme; MRI has no fixed scale).

Usage:
    python analyze_dataset.py <root> --out-dir analysis_out
    python analyze_dataset.py <root> --out-dir analysis_out \
        --seq-table sequences.csv --seq-subject-col Paciente \
        --seq-series-col Serie --seq-class-col "Clase W Final"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import nibabel as nib

from pairs import (find_pairs, load_sequence_table, plane_from_affine,
                   plane_from_name, resolve_sequence)


# ---------------------------------------------------------------------------
# Per-scan analysis
# ---------------------------------------------------------------------------

def analyse_pair(image_path: Path, seg_path: Path,
                 seq_lookup: Optional[dict], subject: str, scan: str) -> dict:
    img = nib.load(str(image_path))
    arr = img.get_fdata(dtype=np.float32)
    zooms = img.header.get_zooms()[:3]
    affine = img.affine

    seg = nib.load(str(seg_path))
    seg_arr = np.asarray(seg.dataobj)
    fg = seg_arr > 0
    n_fg = int(fg.sum())
    labels = sorted(int(v) for v in np.unique(seg_arr) if v != 0)
    voxel_vol = float(np.prod(zooms))                  # mm^3 per voxel

    # intensity within foreground (what normalisation must handle)
    fg_vals = arr[fg] if n_fg > 0 else np.array([0.0], dtype=np.float32)

    seq = resolve_sequence(scan, subject, seq_lookup)
    plane = plane_from_affine(affine, zooms)

    return dict(
        nx=arr.shape[0], ny=arr.shape[1], nz=arr.shape[2],
        sx=round(float(zooms[0]), 4), sy=round(float(zooms[1]), 4),
        sz=round(float(zooms[2]), 4),
        slice_thickness=round(float(max(zooms)), 4),
        plane=plane,
        plane_name=plane_from_name(scan),
        sequence=seq,
        img_min=float(arr.min()), img_max=float(arr.max()),
        img_mean=float(arr.mean()), img_p99=float(np.percentile(arr, 99)),
        fg_voxels=n_fg,
        fg_fraction=round(n_fg / arr.size, 6),
        tumour_volume_mm3=round(n_fg * voxel_vol, 1),
        fg_int_mean=float(fg_vals.mean()), fg_int_std=float(fg_vals.std()),
        fg_int_p01=float(np.percentile(fg_vals, 1)),
        fg_int_p99=float(np.percentile(fg_vals, 99)),
        n_labels=len(labels),
        labels=";".join(map(str, labels)),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_summary(df: pd.DataFrame, out: Path) -> None:
    lines = []
    lines.append(f"labelled scans: {len(df)}")
    lines.append(f"subjects:       {df['subject'].nunique()}")
    lines.append("")
    lines.append("plane (from affine):")
    lines.append(df["plane"].value_counts().to_string())
    lines.append("")
    lines.append("sequence:")
    lines.append(df["sequence"].value_counts().to_string())
    lines.append("")
    lines.append("plane x sequence:")
    lines.append(pd.crosstab(df["sequence"], df["plane"]).to_string())
    lines.append("")
    for col, label in [("nx", "size-X"), ("ny", "size-Y"), ("nz", "size-Z"),
                       ("sx", "spacing-X"), ("sy", "spacing-Y"),
                       ("sz", "spacing-Z"), ("tumour_volume_mm3", "tumour vol mm3"),
                       ("fg_fraction", "fg fraction")]:
        s = df[col]
        lines.append(f"{label:16s} min={s.min():.3g}  med={s.median():.3g}  "
                     f"mean={s.mean():.3g}  max={s.max():.3g}  "
                     f"p05={s.quantile(.05):.3g}  p95={s.quantile(.95):.3g}")
    lines.append("")
    lines.append("intensity (whole image): "
                 f"min={df['img_min'].min():.3g}  max={df['img_max'].max():.3g}  "
                 f"mean-of-means={df['img_mean'].mean():.3g}")
    lines.append("n_labels per scan: " + df["n_labels"].value_counts().to_string().replace("\n", "  "))
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text)
    print(text)


def make_plots(df: pd.DataFrame, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                              # pragma: no cover
        print(f"(skipping plots: {e})")
        return

    plots = out / "plots"
    plots.mkdir(exist_ok=True)

    # histograms
    for col, title in [("nx", "size X"), ("ny", "size Y"), ("nz", "size Z"),
                       ("sx", "spacing X (mm)"), ("sy", "spacing Y (mm)"),
                       ("sz", "spacing Z (mm)"),
                       ("tumour_volume_mm3", "tumour volume (mm^3)"),
                       ("fg_fraction", "foreground fraction"),
                       ("img_max", "image max intensity")]:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.hist(df[col].dropna(), bins=30, color="steelblue", edgecolor="black")
        ax.set_title(title)
        ax.set_ylabel("scans")
        fig.tight_layout()
        fig.savefig(plots / f"hist_{col}.png", dpi=120)
        plt.close(fig)

    # bar charts
    for col, title in [("plane", "acquisition plane"), ("sequence", "sequence")]:
        vc = df[col].value_counts()
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.bar(vc.index.astype(str), vc.values, color="indianred", edgecolor="black")
        ax.set_title(title)
        ax.set_ylabel("scans")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(plots / f"bar_{col}.png", dpi=120)
        plt.close(fig)

    print(f"plots -> {plots}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", type=Path, help="dataset root")
    ap.add_argument("--out-dir", type=Path, default=Path("analysis_out"))
    ap.add_argument("--seq-table", type=Path, default=None,
                    help="clf_perf/combined_reviewed.csv (for true sequence type)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    seq_lookup = None
    if args.seq_table:
        seq_lookup = load_sequence_table(args.seq_table)
        print(f"loaded sequence table: {len(seq_lookup)} entries")

    rows = []
    n_segs = n_missing_img = 0
    for subject, session, scan, image_path, seg_path, seg_source in find_pairs(args.root):
        n_segs += 1
        if image_path is None:                          # mask exists but image excluded/missing
            n_missing_img += 1
            print(f"  [WARN] no image for mask {seg_path} (scan '{scan}')")
            continue
        try:
            info = analyse_pair(image_path, seg_path, seq_lookup, subject, scan)
        except Exception as e:                          # keep going on bad files
            print(f"  [WARN] {seg_path}: {e}")
            continue
        info.update(subject=subject, session=session, scan=scan,
                    seg_source=seg_source,
                    image_path=str(image_path), seg_path=str(seg_path))
        rows.append(info)

    print(f"\nfound {n_segs} masks; {n_missing_img} had no matching image; "
          f"{len(rows)} pairs analysed OK\n")
    if not rows:
        raise SystemExit("no labelled scans found — check the root path / layout")

    df = pd.DataFrame(rows)
    front = ["subject", "session", "scan", "plane", "sequence"]
    df = df[front + [c for c in df.columns if c not in front]]
    df.to_csv(args.out_dir / "per_scan.csv", index=False)
    print(f"per-scan table -> {args.out_dir / 'per_scan.csv'}\n")

    write_summary(df, args.out_dir)
    make_plots(df, args.out_dir)


if __name__ == "__main__":
    main()
