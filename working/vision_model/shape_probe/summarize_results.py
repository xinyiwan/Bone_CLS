#!/usr/bin/env python3
"""Roll every probe_results*.csv under a results tree into one comparison table.

Instead of opening each CSV on the server one at a time, point this at the
directory and read a single wide table: one row per experiment, with overall
accuracy, per-class recall and (optionally) the confusion matrices.

    python summarize_results.py /scratch-shared/xwan1/BONE-AI/results
    python summarize_results.py /scratch-shared/xwan1/BONE-AI/results \
        --confusion --out-dir /scratch-shared/xwan1/BONE-AI/results/_summary

Experiment metadata is read from the CSV columns themselves (shape_set,
background, num_few_shot, difficulty, model_id) rather than guessed from the
file name, because the columns are what the run actually used. The file stem is
kept as `exp` so a row is always traceable back to its file, and the few-shot
suffix convention (`..._fs` = zero-shot prompt file, `..._fs_1` = 1 shot) is
decoded only as a fallback when `num_few_shot` is absent.

Rows whose parsed_label is PARSE_FAILED are excluded from accuracy (they are
counted separately as `n_unparsed`), matching run_shape_probe.py --summarize.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

PARSE_FAILED = "PARSE_FAILED"


def few_shot_from_name(stem: str) -> str:
    """`probe_results_cli_3_s_s_4b_fs_1` -> '1'; `..._fs` -> '0'; else ''."""
    m = re.search(r"_fs(?:_(\d+))?$", stem)
    if not m:
        return ""
    return m.group(1) or "0"


def field(df: pd.DataFrame, col: str) -> str:
    """Single value if the run is homogeneous in `col`, else 'a|b' listing."""
    if col not in df.columns:
        return ""
    vals = sorted({str(v).strip() for v in df[col] if str(v).strip() and str(v) != "nan"})
    return "|".join(vals)


def exp_name(path: Path, root: Path) -> str:
    """Path-based id, not just the stem: several backgrounds each ship a file
    literally called probe_results.csv, so stems collide across directories."""
    rel = path.relative_to(root) if path.is_relative_to(root) else Path(path.name)
    return str(rel.with_suffix(""))


def summarize_one(path: Path, root: Path) -> tuple[dict, pd.DataFrame | None]:
    df = pd.read_csv(path)
    exp = exp_name(path, root)
    if "correct" not in df.columns or "shape" not in df.columns:
        return {"exp": exp, "rel_path": str(path), "note": "not a probe results CSV"}, None

    scored = df[df.get("parsed_label", "") != PARSE_FAILED].copy()
    row: dict = {
        "exp": exp,
        "rel_path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
        "shape_set": field(scored, "shape_set"),
        "background": field(scored, "background"),
        "modality": field(scored, "modality"),
        "difficulty": field(scored, "difficulty"),
        "few_shot": field(scored, "num_few_shot") or few_shot_from_name(path.stem),
        "model_id": field(scored, "model_id"),
        "n": len(scored),
        "n_unparsed": len(df) - len(scored),
    }
    if scored.empty:
        return row, None

    scored["correct"] = scored["correct"].astype(float)
    classes = sorted(scored["shape"].astype(str).unique())
    row["n_classes"] = len(classes)
    row["chance"] = round(1 / len(classes), 3)
    row["accuracy"] = round(scored["correct"].mean(), 4)
    # Macro accuracy = mean of per-class recalls: unlike overall accuracy it is
    # not inflated by a model that collapses onto whichever class is largest.
    per_class = scored.groupby(scored["shape"].astype(str))["correct"].mean()
    row["macro_acc"] = round(per_class.mean(), 4)

    for cls in classes:
        g = scored[scored["shape"].astype(str) == cls]
        row[f"acc_{cls}"] = round(g["correct"].mean(), 4)
        row[f"n_{cls}"] = len(g)

    # A model guessing one label for everything is the single most common
    # failure here, so surface it directly rather than making the reader
    # reconstruct it from the confusion matrix.
    pred = scored.get("parsed_label", pd.Series(dtype=str)).astype(str)
    if len(pred):
        top = pred.value_counts()
        row["top_pred"] = top.index[0]
        row["top_pred_frac"] = round(top.iloc[0] / len(pred), 3)

    cm = pd.crosstab(scored["shape"].astype(str), pred,
                     rownames=["true"], colnames=["pred"])
    return row, cm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path,
                    help="results directory to scan recursively")
    ap.add_argument("--glob", default="**/probe_results*.csv",
                    help="pattern under root (default: %(default)s)")
    ap.add_argument("--confusion", action="store_true",
                    help="also print a confusion matrix per experiment")
    ap.add_argument("--out-dir", type=Path,
                    help="write summary.csv + per-experiment confusion CSVs here")
    ap.add_argument("--sort", default="exp",
                    help="column to sort the table by (default: %(default)s)")
    args = ap.parse_args()

    root = args.root.resolve()
    # Shard files are partial runs of an experiment already represented by the
    # merged CSV; including them would double-count and show fake low-n rows.
    paths = sorted(p for p in root.glob(args.glob) if ".shard" not in p.name)
    if not paths:
        raise SystemExit(f"no files matching {args.glob!r} under {root}")

    rows, matrices = [], {}
    for p in paths:
        try:
            row, cm = summarize_one(p, root)
        except Exception as exc:  # one broken CSV must not kill the whole sweep
            rows.append({"exp": p.stem, "rel_path": str(p), "note": f"ERROR: {exc}"})
            continue
        rows.append(row)
        if cm is not None:
            matrices[row["exp"]] = cm

    table = pd.DataFrame(rows)
    if args.sort in table.columns:
        table = table.sort_values(args.sort, na_position="last")

    lead = [c for c in ("exp", "shape_set", "background", "modality", "difficulty",
                        "few_shot", "n", "n_unparsed", "n_classes", "chance",
                        "accuracy", "macro_acc", "top_pred", "top_pred_frac")
            if c in table.columns]
    acc_cols = sorted(c for c in table.columns if c.startswith("acc_"))
    rest = [c for c in table.columns if c not in lead + acc_cols]
    table = table[lead + acc_cols + rest]

    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(f"{len(paths)} result file(s) under {root}\n")
        print(table.to_string(index=False))

    if args.confusion:
        for exp, cm in matrices.items():
            print(f"\n=== {exp} — confusion (rows = true, cols = predicted) ===")
            print(cm.to_string())

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out = args.out_dir / "summary.csv"
        table.to_csv(out, index=False)
        for exp, cm in matrices.items():
            cm.to_csv(args.out_dir / f"confusion_{exp.replace('/', '__')}.csv")
        print(f"\nwrote {out} and {len(matrices)} confusion matrix file(s) "
              f"to {args.out_dir}")


if __name__ == "__main__":
    main()
