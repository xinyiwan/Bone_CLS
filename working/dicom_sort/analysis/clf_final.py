"""Summarise the transition from original CLF predictions to reviewed final labels.

Builds a combined 3-part class string (W-FS-C, e.g. ``T1W-noFS-noC``) for both the
predicted side (``Predicción Clases W/FS/C``) and the final side
(``Clase W Final``, ``Clase FS Final``, ``Clase C Final``), then renders a Sankey
diagram of original -> final flow.

Usage:
    python clf_final.py /path/to/labels.csv
    python clf_final.py /path/to/labels.csv --output transitions.html
    python clf_final.py /path/to/labels.csv --no-unchanged   # hide same-class flows
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


PRED_COLS = ("Predicción Clases W", "Predicción Clases FS", "Predicción Clases C")
FINAL_COLS = ("Clase W Final", "Clase FS Final", "Clase C Final")


def _norm(value: object) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    return "" if s.lower() == "nan" else s


def _fmt_fs(value: str) -> str | None:
    v = _norm(value)
    if v == "Y":
        return "FS"
    if v == "N":
        return "noFS"
    if v in ("-", ""):
        return None
    return v


def _fmt_c(value: str) -> str | None:
    v = _norm(value)
    if v == "Y":
        return "C"
    if v == "N":
        return "noC"
    if v in ("-", ""):
        return None
    return v


def combine_class(w: object, fs: object, c: object) -> str:
    """Join W / FS / C into the spec format, e.g. ``T1W-noFS-noC`` or ``Other``."""
    w_s = _norm(w)
    if not w_s or w_s == "-":
        return "Unlabeled"
    parts = [w_s]
    fs_p = _fmt_fs(fs)
    if fs_p is not None:
        parts.append(fs_p)
    c_p = _fmt_c(c)
    if c_p is not None:
        parts.append(c_p)
    return "-".join(parts)


def build_transition_table(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in PRED_COLS + FINAL_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    original = df.apply(
        lambda r: combine_class(r[PRED_COLS[0]], r[PRED_COLS[1]], r[PRED_COLS[2]]),
        axis=1,
    )
    final = df.apply(
        lambda r: combine_class(r[FINAL_COLS[0]], r[FINAL_COLS[1]], r[FINAL_COLS[2]]),
        axis=1,
    )
    counts = (
        pd.DataFrame({"original": original, "final": final})
        .value_counts()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    return counts


def render_sankey(transitions: pd.DataFrame, output: Path, drop_unchanged: bool) -> None:
    data = transitions.copy()
    if drop_unchanged:
        data = data[data["original"] != data["final"]]
    if data.empty:
        raise ValueError("No transitions to plot (table is empty after filtering).")

    originals = [f"{lbl}  (orig)" for lbl in sorted(data["original"].unique())]
    finals = [f"{lbl}  (final)" for lbl in sorted(data["final"].unique())]
    nodes = originals + finals
    idx = {name: i for i, name in enumerate(nodes)}

    sources = [idx[f"{o}  (orig)"] for o in data["original"]]
    targets = [idx[f"{f}  (final)"] for f in data["final"]]
    values = data["count"].tolist()

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(label=nodes, pad=15, thickness=18),
            link=dict(source=sources, target=targets, value=values),
        )
    )
    title = "Original (CLF prediction) → Final (reviewed)"
    if drop_unchanged:
        title += " — unchanged flows hidden"
    fig.update_layout(title_text=title, font_size=12)
    fig.write_html(str(output))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", type=Path, help="Path to the labels CSV.")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output HTML file (default: alongside the CSV as <csv>_sankey.html).",
    )
    parser.add_argument(
        "--no-unchanged",
        action="store_true",
        help="Hide flows where original == final to highlight only the changes.",
    )
    parser.add_argument(
        "--save-table",
        type=Path,
        default=None,
        help="Optional path to also dump the transition counts as CSV.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv, dtype=str).fillna("")
    if "viewed" not in df.columns:
        raise KeyError("Expected a 'viewed' column to filter reviewed rows.")
    before = len(df)
    df = df[df["viewed"].str.strip().str.upper() == "X"].reset_index(drop=True)
    print(f"Reviewed rows kept: {len(df)} / {before}")
    transitions = build_transition_table(df)

    total = int(transitions["count"].sum())
    changed = int(transitions.loc[transitions["original"] != transitions["final"], "count"].sum())
    print(f"Rows analysed: {total}")
    print(f"Unchanged:     {total - changed}")
    print(f"Changed:       {changed}")
    print("\nTop transitions:")
    with pd.option_context("display.max_rows", 30, "display.max_colwidth", 40):
        print(transitions.head(30).to_string(index=False))

    output = args.output or args.csv.with_name(f"{args.csv.stem}_sankey.html")
    render_sankey(transitions, output, drop_unchanged=args.no_unchanged)
    print(f"\nSankey written to: {output}")

    if args.save_table is not None:
        transitions.to_csv(args.save_table, index=False)
        print(f"Transition table written to: {args.save_table}")


if __name__ == "__main__":
    main()
