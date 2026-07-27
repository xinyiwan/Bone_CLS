"""
Local review UI for MedGemma inference results (no web framework needed).

Reads inference_results.csv (one row per image, produced by run_medgemma.py
--mode infer) and serves a small site to eyeball, per subject and feature:
the crop image, the model's prediction + reason, and the ground-truth label.

    python review_server.py --results inference_results.csv --port 8000
    # then open http://localhost:8000

Only images referenced in the CSV are served (path whitelist), and the paths
must be reachable from wherever you run this (they're absolute in the pipeline
output, e.g. /output/.../shape/T1W_axial_10.png). Stdlib only: no pip installs
beyond pandas, which the pilot already uses.
"""

from __future__ import annotations

import argparse
import html
import mimetypes
import urllib.parse
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Populated in main(): case_id -> feature_name -> list[row-dict]; and the set of
# image paths we're allowed to serve.
DATA: Dict[str, Dict[str, List[dict]]] = {}
IMAGE_WHITELIST: set = set()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load(results_csv: Path) -> None:
    df = pd.read_csv(results_csv, dtype=str).fillna("")
    for col in ("case_id", "feature_name", "image_path", "parsed_label"):
        if col not in df.columns:
            raise SystemExit(f"{results_csv}: missing required column {col!r} (have {list(df.columns)})")

    for _, r in df.iterrows():
        row = {k: r.get(k, "") for k in (
            "plane", "modality", "image_path", "parsed_label",
            "reason", "ground_truth_label", "correct", "raw_output")}
        DATA.setdefault(r["case_id"], {}).setdefault(r["feature_name"], []).append(row)
        if row["image_path"]:
            IMAGE_WHITELIST.add(row["image_path"])


def majority(labels: List[str]) -> str:
    valid = [l for l in labels if l and l != "PARSE_FAILED"]
    if not valid:
        return "PARSE_FAILED"
    counts = Counter(valid)
    top = max(counts.values())
    for l in valid:                       # first-seen wins ties (matches run_medgemma.aggregate)
        if counts[l] == top:
            return l
    return valid[0]


def feature_gt(rows: List[dict]) -> str:
    for r in rows:
        if r["ground_truth_label"]:
            return r["ground_truth_label"]
    return "unknown"


def has_gt(gt: str) -> bool:
    return gt.strip().lower() not in {"", "unknown"}


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 24px; color: #1a1a1a; }}
 a {{ color: #2557a7; text-decoration: none; }} a:hover {{ text-decoration: underline; }}
 h1 {{ font-size: 20px; }} h2 {{ font-size: 17px; margin-top: 28px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
 table {{ border-collapse: collapse; }} td, th {{ padding: 6px 12px; border-bottom: 1px solid #eee; text-align: left; }}
 .grid {{ display: flex; flex-wrap: wrap; gap: 16px; }}
 .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 10px; width: 240px; }}
 .card img {{ width: 220px; height: 220px; object-fit: contain; background: #000; border-radius: 4px; }}
 .meta {{ font-size: 12px; color: #666; margin: 6px 0 2px; }}
 .reason {{ font-size: 13px; color: #333; margin-top: 4px; }}
 .badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 12px; font-weight: 600; }}
 .ok {{ background: #e3f4e4; color: #1a7f37; }}
 .bad {{ background: #fce8e6; color: #c5221f; }}
 .neutral {{ background: #eef; color: #333; }}
 .fail {{ background: #fff3cd; color: #856404; }}
</style></head><body>{body}</body></html>"""


def esc(s) -> str:
    return html.escape(str(s))


def img_url(path: str) -> str:
    return "/img?path=" + urllib.parse.quote(path, safe="")


def pred_badge(pred: str, gt: str) -> str:
    if pred == "PARSE_FAILED":
        return f'<span class="badge fail">{esc(pred)}</span>'
    if not has_gt(gt):
        return f'<span class="badge neutral">{esc(pred)}</span>'
    cls = "ok" if pred.strip().lower() == gt.strip().lower() else "bad"
    return f'<span class="badge {cls}">{esc(pred)}</span>'


def index_html() -> str:
    rows = []
    for case_id in sorted(DATA):
        feats = DATA[case_id]
        n_img = sum(len(v) for v in feats.values())
        n_scored = n_correct = 0
        for frows in feats.values():
            for r in frows:
                if has_gt(r["ground_truth_label"]):
                    n_scored += 1
                    if r["correct"].strip().lower() in {"true", "1"}:
                        n_correct += 1
        acc = f"{n_correct}/{n_scored}" if n_scored else "—"
        rows.append(
            f'<tr><td><a href="/subject?id={urllib.parse.quote(case_id)}">{esc(case_id)}</a></td>'
            f"<td>{len(feats)}</td><td>{n_img}</td><td>{acc}</td></tr>"
        )
    body = (
        f"<h1>MedGemma review — {len(DATA)} subject(s)</h1>"
        "<p>Per-image correct / scored shown where ground truth is known.</p>"
        "<table><tr><th>subject</th><th>features</th><th>images</th><th>correct/scored</th></tr>"
        + "".join(rows) + "</table>"
    )
    return PAGE.format(title="MedGemma review", body=body)


def subject_html(case_id: str) -> str:
    feats = DATA[case_id]
    sections = [f'<h1><a href="/">&larr; all subjects</a> &nbsp; {esc(case_id)}</h1>']
    for feature in sorted(feats):
        rows = feats[feature]
        gt = feature_gt(rows)
        maj = majority([r["parsed_label"] for r in rows])
        gt_disp = pred_badge(maj, gt) if has_gt(gt) else '<span class="badge neutral">unknown</span>'
        header = (
            f"<h2>{esc(feature)} "
            f'&nbsp;<span class="meta">ground truth: <b>{esc(gt)}</b> &nbsp;|&nbsp; '
            f"majority prediction: </span>{pred_badge(maj, gt)}</h2>"
        )
        cards = []
        for r in rows:
            reason = esc(r["reason"]) if r["reason"] else '<i style="color:#999">(no reason)</i>'
            cards.append(
                '<div class="card">'
                f'<img src="{img_url(r["image_path"])}" loading="lazy">'
                f'<div class="meta">{esc(r["modality"])} · {esc(r["plane"])}</div>'
                f'<div>prediction: {pred_badge(r["parsed_label"], gt)}</div>'
                f'<div class="reason">{reason}</div>'
                "</div>"
            )
        sections.append(header + '<div class="grid">' + "".join(cards) + "</div>")
    return PAGE.format(title=f"{case_id} — review", body="".join(sections))


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"      # keep-alive + reliable Content-Length framing

    def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)

        if parsed.path == "/":
            self._send(index_html().encode(), "text/html; charset=utf-8")
        elif parsed.path == "/subject":
            cid = (qs.get("id") or [""])[0]
            if cid in DATA:
                self._send(subject_html(cid).encode(), "text/html; charset=utf-8")
            else:
                self._send(b"unknown subject", "text/plain", 404)
        elif parsed.path == "/img":
            path = (qs.get("path") or [""])[0]
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
    ap.add_argument("--results", type=Path, default=Path("inference_results.csv"),
                    help="per-image results CSV from run_medgemma.py --mode infer")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    load(args.results)
    n_img = sum(len(v) for feats in DATA.values() for v in feats.values())
    print(f"loaded {len(DATA)} subject(s), {n_img} image row(s) from {args.results}")
    print(f"serving at http://{args.host}:{args.port}  (Ctrl-C to stop)")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
