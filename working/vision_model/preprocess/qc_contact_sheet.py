"""
Sanity-check utility: sample N images from the metadata CSV and render a grid
of thumbnails so you can eyeball a batch of extracted crops at once.

Uses the overlay image when present (falls back to the plain crop), so you see
the mask boundary during QC.

Usage:
    python qc_contact_sheet.py ./out/metadata.csv --n 24 --out contact_sheet.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / shared server
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def _overlay_variant(path: Path) -> Path:
    ov = path.with_name(path.stem + "_overlay" + path.suffix)
    return ov if ov.exists() else path


def build_contact_sheet(metadata_csv: Path, n: int, out_path: Path, seed: int = 0) -> None:
    df = pd.read_csv(metadata_csv)
    # explode the ';'-joined image lists to one image per cell
    rows = []
    for _, r in df.iterrows():
        for p in str(r["image_paths"]).split(";"):
            if p:
                rows.append((r["case_id"], r["feature_name"], p))
    if not rows:
        raise SystemExit("no images found in metadata")

    flat = pd.DataFrame(rows, columns=["case_id", "feature_name", "path"])
    sample = flat.sample(min(n, len(flat)), random_state=seed)

    cols = int(math.ceil(math.sqrt(len(sample))))
    rows_n = int(math.ceil(len(sample) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 2, rows_n * 2.2))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for ax, (_, rec) in zip(axes, sample.iterrows()):
        img_path = _overlay_variant(Path(rec["path"]))
        try:
            ax.imshow(plt.imread(str(img_path)), cmap="gray")
        except Exception as e:  # noqa: BLE001
            ax.text(0.5, 0.5, f"missing\n{e}", ha="center", va="center", fontsize=6)
        ax.set_title(f"{rec['case_id']}\n{rec['feature_name']}", fontsize=6)
        ax.axis("off")
    for ax in axes[len(sample):]:
        ax.axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"wrote {out_path} ({len(sample)} thumbnails)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("metadata_csv", type=Path)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", type=Path, default=Path("contact_sheet.png"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    build_contact_sheet(args.metadata_csv, args.n, args.out, args.seed)


if __name__ == "__main__":
    main()
