"""
Local review UI for the shape probe (no web framework needed).

Deliberately FLAT, unlike medgemma_pilot/review_server.py. That one groups by
subject -> feature because the real experiment's unit of truth is the lesion and
its per-image predictions have to be majority-voted. Here every image carries its
own ground truth (the shape we drew), so there is nothing to aggregate: the
question is just "what fraction of images did it name correctly", plus the
ability to click through the ones it got wrong.

    python review_server.py --results probe_results.csv --port 8000
    # then open http://localhost:8000

Pass several CSVs (e.g. shard files, or one per background condition) and they
are concatenated -- the breakdowns will then split by condition.

Only images referenced in the CSV are served (path whitelist), and the paths must
be reachable from wherever you run this. Stdlib only, beyond pandas.
"""

from __future__ import annotations

import argparse
import html
import mimetypes
import urllib.parse
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

PAGE_SIZE = 60  # images per gallery page; a full run is thousands of tiny PNGs

ROWS: List[dict] = []
IMAGE_WHITELIST: set = set()
RUN_INFO: Dict[str, object] = {}

ROW_COLS = (
    "case_id", "feature_name", "modality", "plane", "image_path", "background",
    "shape_set", "difficulty", "num_few_shot",
    "radius_px", "rotation_deg", "parsed_label", "reason", "shape", "correct",
    "model_id", "input_text", "thinking", "raw_output",
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load(results_csvs: List[Path]) -> None:
    frames = []
    for p in results_csvs:
        df = pd.read_csv(p, dtype=str).fillna("")
        # Guard against being handed the BUILD metadata instead of the results:
        # it has image_path and shape but no prediction, so every stat would be
        # silently empty. Name the mistake rather than serving a blank page.
        if "parsed_label" not in df.columns:
            raise SystemExit(
                f"{p} has no 'parsed_label' column.\n"
                "This looks like shape_metadata.csv (the build_shapes.py output), not the "
                "inference results. Point --results at run_shape_probe.py --mode infer --out."
            )
        for col in ("image_path", "shape"):
            if col not in df.columns:
                raise SystemExit(f"{p}: missing required column {col!r} (have {list(df.columns)})")
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    for _, r in df.iterrows():
        row = {k: r.get(k, "") for k in ROW_COLS}
        ROWS.append(row)
        if row["image_path"]:
            IMAGE_WHITELIST.add(row["image_path"])

    def distinct(col: str) -> List[str]:
        return sorted({str(v).strip() for v in df[col] if str(v).strip()}) if col in df.columns else []

    RUN_INFO.update({
        "models": distinct("model_id"),
        "backgrounds": distinct("background"),
        "n_rows": len(df),
        "n_files": len(results_csvs),
        # Which vocabulary this run used, so the header describes the actual task
        # instead of hardcoding one shape set's labels.
        "shape_sets": distinct("shape_set"),
    })


def classes() -> List[str]:
    """The classes this run actually used, read off the ground truth.

    NOT a hardcoded list: `build_shapes.py --skip-shapes` can leave classes out,
    so the number of alternatives -- and therefore the chance level -- is a
    property of the loaded rows. Hardcoding 4 shapes made every clinical run
    display the wrong baseline."""
    return sorted({str(r["shape"]).strip() for r in ROWS if str(r["shape"]).strip()})


def chance_pct() -> Optional[float]:
    """Chance accuracy in percent, or None when there is nothing to infer it from."""
    n = len(classes())
    return 100.0 / n if n else None


def is_scored(r: dict) -> bool:
    """PARSE_FAILED rows are excluded from accuracy but still shown, so a low
    score is never quietly a parsing problem."""
    return r["parsed_label"] not in {"", "PARSE_FAILED", "ERROR"}


def is_correct(r: dict) -> bool:
    return is_scored(r) and r["parsed_label"].strip().lower() == r["shape"].strip().lower()


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def acc_str(correct: int, scored: int) -> str:
    return f"{correct}/{scored} ({correct / scored * 100:.1f}%)" if scored else "—"


def breakdown(key: str) -> Dict[str, list]:
    out: Dict[str, list] = {}
    for r in ROWS:
        if not is_scored(r):
            continue
        k = r.get(key) or "?"
        out.setdefault(k, [0, 0])
        out[k][0] += 1 if is_correct(r) else 0
        out[k][1] += 1
    return out


def overall() -> tuple:
    scored = [r for r in ROWS if is_scored(r)]
    return sum(is_correct(r) for r in scored), len(scored)


def confusion() -> Dict[str, Counter]:
    m: Dict[str, Counter] = {s: Counter() for s in SHAPES}
    for r in ROWS:
        if not is_scored(r):
            continue
        m.setdefault(r["shape"], Counter())[r["parsed_label"]] += 1
    return m


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 24px; color: #1a1a1a; }}
 a {{ color: #2557a7; text-decoration: none; }} a:hover {{ text-decoration: underline; }}
 h1 {{ font-size: 20px; }} h2 {{ font-size: 17px; margin-top: 28px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
 table {{ border-collapse: collapse; }} td, th {{ padding: 6px 12px; border-bottom: 1px solid #eee; text-align: left; }}
 th {{ font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: .03em; }}
 .grid {{ display: flex; flex-wrap: wrap; gap: 14px; }}
 .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 8px; width: 190px; }}
 .card img {{ width: 174px; height: 174px; object-fit: contain; background: #000; border-radius: 4px; }}
 .meta {{ font-size: 12px; color: #666; margin: 5px 0 2px; }}
 .reason {{ font-size: 12px; color: #333; margin-top: 4px; }}
 .badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 12px; font-weight: 600; }}
 .ok {{ background: #e3f4e4; color: #1a7f37; }}
 .bad {{ background: #fce8e6; color: #c5221f; }}
 .neutral {{ background: #eef; color: #333; }}
 .fail {{ background: #fff3cd; color: #856404; }}
 .summary {{ background: #f6f8fa; border: 1px solid #ddd; border-radius: 8px; padding: 12px 18px; margin-bottom: 20px; }}
 .summary h1 {{ margin: 0 0 4px; }}
 .runinfo {{ margin-top: 10px; padding-top: 8px; border-top: 1px solid #e3e3e3; font-size: 13px; }}
 .runinfo code {{ background: #eceff4; padding: 1px 5px; border-radius: 3px; font-size: 12px; }}
 .statgrid {{ display: flex; gap: 40px; flex-wrap: wrap; margin-top: 8px; }}
 .statgrid h3 {{ font-size: 13px; margin: 0 0 4px; color: #666; text-transform: uppercase; letter-spacing: .03em; }}
 .filters {{ margin: 14px 0; font-size: 13px; }}
 .filters a {{ display: inline-block; padding: 3px 10px; border: 1px solid #ccd; border-radius: 12px; margin: 2px 4px 2px 0; }}
 .filters a.on {{ background: #2557a7; color: #fff; border-color: #2557a7; }}
 .diag {{ background: #eef6ee; font-weight: 600; }}
 .bar {{ height: 8px; background: #2557a7; border-radius: 4px; display: inline-block; vertical-align: middle; }}
 .chance {{ color: #999; font-size: 12px; }}
 details.raw {{ margin-top: 5px; }}
 details.raw summary {{ cursor: pointer; font-size: 12px; color: #2557a7; }}
 details.raw pre {{ white-space: pre-wrap; word-break: break-word; font-size: 11px; line-height: 1.4;
                   max-height: 300px; overflow: auto; background: #f7f7f7; padding: 8px; border-radius: 4px; margin: 6px 0 0; }}
</style></head><body>{body}</body></html>"""


def esc(s) -> str:
    return html.escape(str(s))


def img_url(path: str) -> str:
    return "/img?path=" + urllib.parse.quote(path, safe="")


def pred_badge(r: dict) -> str:
    pred = r["parsed_label"]
    if not is_scored(r):
        return f'<span class="badge fail">{esc(pred or "no answer")}</span>'
    return f'<span class="badge {"ok" if is_correct(r) else "bad"}">{esc(pred)}</span>'


def _stat_table(title: str, table: Dict[str, list]) -> str:
    if not table:
        return ""
    rows = "".join(
        f'<tr><td><a href="/images?{title}={urllib.parse.quote(k)}">{esc(k)}</a></td>'
        f"<td>{acc_str(c, n)}</td></tr>"
        for k, (c, n) in sorted(table.items())
    )
    return f"<div><h3>by {esc(title)}</h3><table>{rows}</table></div>"


def run_info_html() -> str:
    models = ", ".join(RUN_INFO.get("models") or []) or "unknown"
    bgs = ", ".join(RUN_INFO.get("backgrounds") or []) or "unknown"
    n_files = int(RUN_INFO.get("n_files", 1))
    files = f' &nbsp;·&nbsp; <b>files:</b> {n_files} combined' if n_files > 1 else ""
    sets = ", ".join(RUN_INFO.get("shape_sets") or [])
    shape_set = f' &nbsp;·&nbsp; <b>shape set:</b> {esc(sets)}' if sets else ""
    return ('<div class="runinfo">'
            f'<b>model:</b> <code>{esc(models)}</code>'
            f' &nbsp;·&nbsp; <b>background:</b> {esc(bgs)}'
            f' &nbsp;·&nbsp; <b>task:</b> name the red outline'
            f"{shape_set}{files}</div>")


def confusion_html() -> str:
    preds = sorted({r["parsed_label"] for r in ROWS if is_scored(r)})
    m = confusion()
    head = "".join(f"<th>{esc(p)}</th>" for p in preds)
    body = ""
    for true_shape in sorted(m):
        cells = ""
        for p in preds:
            n = m[true_shape][p]
            cls = ' class="diag"' if p.strip().lower() == true_shape.strip().lower() else ""
            cells += f"<td{cls}>{n or ''}</td>"
        body += f"<tr><th>{esc(true_shape)}</th>{cells}</tr>"
    return ("<h2>Confusion matrix</h2>"
            '<div class="meta">rows = shape drawn, columns = shape predicted</div>'
            f"<table><tr><th></th>{head}</tr>{body}</table>")


def distribution_html() -> str:
    """Prediction distribution. A model that just always says one label lands at
    the chance level too, so this is what separates 'chance' from 'partial
    ability'."""
    scored = [r for r in ROWS if is_scored(r)]
    if not scored:
        return ""
    counts = Counter(r["parsed_label"] for r in scored)
    n = len(scored)
    rows = ""
    for label, c in counts.most_common():
        pct = c / n * 100
        rows += (f"<tr><td>{esc(label)}</td><td>{c}</td><td>{pct:.1f}%</td>"
                 f'<td><span class="bar" style="width:{pct * 2:.0f}px"></span></td></tr>')
    ch = chance_pct()
    baseline = f"the {ch:.0f}%" if ch is not None else "the accuracy above"
    return ("<h2>Prediction distribution</h2>"
            f'<div class="meta">if this collapses onto one label, {baseline} is guessing, '
            "not partial perception</div>"
            f"<table><tr><th>predicted</th><th>n</th><th>share</th><th></th></tr>{rows}</table>")


def summary_html() -> str:
    c, n = overall()
    n_unscored = len(ROWS) - n
    labels = classes()
    chance = chance_pct()
    verdict = ""
    if n and chance is not None:
        pct = c / n * 100
        verdict = (' <span class="badge ok">above chance</span>' if pct > chance + 5
                   else ' <span class="badge bad">at chance — no evidence the overlay is seen</span>')
    unscored = (f'<div class="meta">{n_unscored} row(s) excluded: no parseable answer</div>'
                if n_unscored else "")
    chance_txt = (f"chance = {chance:.0f}% ({len(labels)} classes: {esc(', '.join(labels))})"
                  " · correct / scored images"
                  if chance is not None else "correct / scored images")
    return ('<div class="summary">'
            f"<h1>Overall accuracy: {acc_str(c, n)}{verdict}</h1>"
            f'<span class="chance">{chance_txt}</span>'
            + unscored + run_info_html()
            + '<div class="statgrid">'
            + _stat_table("shape", breakdown("shape"))
            # difficulty is the psychometric curve and num_few_shot the
            # zero- vs few-shot contrast -- the two comparisons the probe exists
            # to make, so they sit next to the class breakdown, not below it.
            + _stat_table("difficulty", breakdown("difficulty"))
            + _stat_table("num_few_shot", breakdown("num_few_shot"))
            + _stat_table("background", breakdown("background"))
            + _stat_table("modality", breakdown("modality"))
            + _stat_table("plane", breakdown("plane"))
            + "</div></div>")


def index_html() -> str:
    body = (summary_html()
            + '<div class="filters"><a href="/images">browse all images</a>'
              '<a href="/images?result=wrong">only wrong</a>'
              '<a href="/images?result=correct">only correct</a>'
              '<a href="/images?result=unparsed">unparseable answers</a></div>'
            + confusion_html() + distribution_html())
    return PAGE.format(title="Shape probe review", body=body)


# Facets that both get an accuracy table on the summary page and can filter the
# gallery. One list, so a table's link can never point at a facet the filter
# ignores -- add a column here and both sides pick it up.
FACETS = ("shape", "difficulty", "num_few_shot", "background", "modality", "plane")


def filter_rows(result: str, facets: Dict[str, str]) -> List[dict]:
    out = ROWS
    if result == "wrong":
        out = [r for r in out if is_scored(r) and not is_correct(r)]
    elif result == "correct":
        out = [r for r in out if is_correct(r)]
    elif result == "unparsed":
        out = [r for r in out if not is_scored(r)]
    for key, val in facets.items():
        if val:
            out = [r for r in out if (r.get(key) or "?") == val]
    return out


def images_html(rows: List[dict], page: int, query: Dict[str, str]) -> str:
    total = len(rows)
    pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    page = max(1, min(page, pages))
    chunk = rows[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]

    def link(**over) -> str:
        q = {**query, **over}
        return "/images?" + urllib.parse.urlencode({k: v for k, v in q.items() if v})

    tabs = "".join(
        f'<a class="{"on" if query.get("result", "") == v else ""}" href="{link(result=v, page="")}">{lbl}</a>'
        for v, lbl in (("", "all"), ("correct", "correct"), ("wrong", "wrong"), ("unparsed", "no answer"))
    )
    nav = ""
    if pages > 1:
        prev = f'<a href="{link(page=str(page - 1))}">&larr; prev</a>' if page > 1 else ""
        nxt = f'<a href="{link(page=str(page + 1))}">next &rarr;</a>' if page < pages else ""
        nav = f'<div class="filters">{prev} page {page}/{pages} {nxt}</div>'

    cards = []
    for r in chunk:
        reason = esc(r["reason"]) if r["reason"] else '<i style="color:#999">(no reason)</i>'
        inp = (f"<details class='raw'><summary>input / prompt</summary><pre>{esc(r['input_text'])}</pre></details>"
               if r.get("input_text") else "")
        think = (f"<details class='raw'><summary>thinking</summary><pre>{esc(r['thinking'])}</pre></details>"
                 if r.get("thinking") else "")
        raw = (f"<details class='raw'><summary>raw output</summary><pre>{esc(r['raw_output'])}</pre></details>"
               if r.get("raw_output") else "")
        cards.append(
            '<div class="card">'
            f'<img src="{img_url(r["image_path"])}" loading="lazy">'
            f'<div class="meta">{esc(r["case_id"])} · {esc(r["modality"])} · {esc(r["plane"])}</div>'
            f'<div>drew: <b>{esc(r["shape"])}</b> &nbsp; said: {pred_badge(r)}</div>'
            f'<div class="reason">{reason}</div>{inp}{think}{raw}</div>'
        )

    body = (f'<h1><a href="/">&larr; summary</a> &nbsp; {total} image(s)</h1>'
            f'<div class="filters">{tabs}</div>{nav}'
            '<div class="grid">' + "".join(cards) + "</div>" + nav)
    return PAGE.format(title=f"Shape probe — {total} images", body=body)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)

        def q(name: str) -> str:
            return (qs.get(name) or [""])[0]

        if parsed.path == "/":
            self._send(index_html().encode(), "text/html; charset=utf-8")
        elif parsed.path == "/images":
            rows = filter_rows(q("result"), {k: q(k) for k in FACETS})
            try:
                page = int(q("page") or 1)
            except ValueError:
                page = 1
            query = {k: q(k) for k in ("result", *FACETS)}
            self._send(images_html(rows, page, query).encode(), "text/html; charset=utf-8")
        elif parsed.path == "/img":
            path = q("path")
            if path not in IMAGE_WHITELIST:        # only serve images named in the CSV
                self._send(b"forbidden", "text/plain", 403)
                return
            p = Path(path)
            if not p.is_file():
                self._send(b"image not found on disk", "text/plain", 404)
                return
            ctype = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
            self._send(p.read_bytes(), ctype)
        else:
            self._send(b"not found", "text/plain", 404)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", type=Path, nargs="+", required=True,
                    help="per-image results CSV(s) from run_shape_probe.py --mode infer; "
                         "pass several (shards, or one per background condition) to combine")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    load(args.results)
    c, n = overall()
    print(f"loaded {len(ROWS)} image row(s) from {len(args.results)} file(s)")
    ch = chance_pct()
    print(f"overall accuracy: {acc_str(c, n)}"
          + (f"  (chance = {ch:.0f}%, {len(classes())} classes)" if ch is not None else ""))
    print(f"serving at http://{args.host}:{args.port}  (Ctrl-C to stop)")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
