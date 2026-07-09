"""
Batch CLI driver: extract 2D crops for every subject x feature.

Subjects come from --subject-list (one id per line) or by scanning --data-root
for immediate subdirectories. Missing modalities/masks are logged and skipped;
one bad subject never kills the batch.

Usage (single test subject first):
    python run.py --data-root /data --out-root ./out \
        --config feature_config.yaml --subjects SUBJ001 --overlay

Full run:
    python run.py --data-root /data --out-root ./out \
        --config feature_config.yaml --subject-list subjects.txt

Then QC:
    python qc_contact_sheet.py ./out/metadata.csv --n 24 --out contact_sheet.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

from config import load_feature_config
from outputs import MetadataWriter
from pipeline import PipelineOptions, process_subject


def discover_subjects(data_root: Path) -> List[str]:
    return sorted(p.name for p in data_root.iterdir() if p.is_dir())


def load_subject_list(path: Path) -> List[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True, help="feature mapping (.yaml/.csv)")
    ap.add_argument("--subject-list", type=Path, help="file with one subject id per line")
    ap.add_argument("--subjects", nargs="*", help="explicit subject ids (overrides scan/list)")
    ap.add_argument("--metadata", type=Path, help="metadata CSV path (default out-root/metadata.csv)")
    # pipeline options
    ap.add_argument("--out-size", type=int, default=128)
    ap.add_argument("--norm", choices=["minmax", "zscore"], default="minmax")
    ap.add_argument("--pad-mode", choices=["clip", "pad"], default="clip")
    ap.add_argument("--crop-mode", choices=["bbox", "masked"], default="bbox",
                    help="bbox=keep surrounding tissue; masked=lesion only (config can override per-feature)")
    ap.add_argument("--mask-dilate-px", type=int, default=0,
                    help="masked mode: keep a rim of tissue this many px around the lesion")
    ap.add_argument("--no-foreground-norm", action="store_true", help="use full FOV for percentiles")
    ap.add_argument("--overlay", action="store_true", help="also write mask-contour QC images")
    ap.add_argument("--img-pattern", default="{modality}.nii.gz")
    ap.add_argument("--seg-pattern", default="{modality}_seg.nii.gz")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("preprocess")

    specs = load_feature_config(args.config)
    log.info("loaded %d feature(s): %s", len(specs), ", ".join(s.name for s in specs))

    if args.subjects:
        subjects = args.subjects
    elif args.subject_list:
        subjects = load_subject_list(args.subject_list)
    else:
        subjects = discover_subjects(args.data_root)
    log.info("%d subject(s) to process", len(subjects))

    opt = PipelineOptions(
        out_size=(args.out_size, args.out_size),
        norm_method=args.norm,
        pad_mode=args.pad_mode,
        crop_mode=args.crop_mode,
        mask_dilate_px=args.mask_dilate_px,
        foreground_only=not args.no_foreground_norm,
        overlay=args.overlay,
        img_pattern=args.img_pattern,
        seg_pattern=args.seg_pattern,
    )

    meta_path = args.metadata or (args.out_root / "metadata.csv")
    with MetadataWriter(meta_path) as writer:
        for i, subject in enumerate(subjects, 1):
            log.info("[%d/%d] %s", i, len(subjects), subject)
            process_subject(subject, specs, args.data_root, args.out_root, opt, writer)

    log.info("done -> %s", meta_path)


if __name__ == "__main__":
    main()
