"""
Eyeball QC: tile a sample of the generated shape images into one PNG, so you can
confirm the shapes are visible, centred on the lesion and not clipped BEFORE
spending GPU time on inference.

    python preview.py --metadata /results/shape_probe/mri/shape_metadata.csv \
        --out /results/shape_probe/mri/preview.png --n 24
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def contact_sheet(metadata: Path, out: Path, n: int = 24, cols: int = 6,
                  cell: int = 160, seed: int = 0) -> None:
    df = pd.read_csv(metadata)
    sample = df.sample(min(n, len(df)), random_state=seed)

    rows = int(np.ceil(len(sample) / cols))
    sheet = np.zeros((rows * cell, cols * cell, 3), dtype=np.uint8)

    for i, (_, r) in enumerate(sample.iterrows()):
        img = cv2.imread(str(r["image_path"]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.resize(img, (cell, cell), interpolation=cv2.INTER_NEAREST)
        cv2.putText(img, str(r["shape"]), (4, cell - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 0), 1, cv2.LINE_AA)
        y, x = (i // cols) * cell, (i % cols) * cell
        sheet[y:y + cell, x:x + cell] = img

    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    print(f"wrote {out}  ({len(sample)} tiles)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    contact_sheet(args.metadata, args.out, args.n, args.cols, seed=args.seed)


if __name__ == "__main__":
    main()
