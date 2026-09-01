"""
Human re-annotation UI for the preprocessed feature crops (no web framework).

Unlike medgemma_pilot/review_server.py and shape_probe/review_server.py -- which
audit a MODEL run and are read-only -- this one is driven by the preprocess
pipeline's own metadata.csv and WRITES back: an annotator walks the crops and
records, per image and per subject, whether the existing label is right.

    python annotation_server.py \
        --metadata /output/preprocess/all_feature/metadata.csv \
        --images-root /projects/prjs1779/BONE-AI/output/preprocess/all_feature \
        --feature-config ../preprocess/feature_config.yaml \
        --labels-dir /output/vision_model/label_out/jsons \
        --out annotations.csv --annotator kira
    # then open http://localhost:8000

Label vocabulary: feature_config.yaml gives feature -> assessment_key only, so
the option list per feature is the set of values actually seen under that key
across --labels-dir. That guarantees a corrected label compares directly against
metadata.csv's `ground_truth_label` -- no casing/mapping layer. (Do NOT source it
from medgemma_pilot/feature_prompts.yaml: those `label_options` are the model's
prompt vocabulary and differ in case and wording from the assessment JSON.)

Keys under `imaging_features` whose value is a LIST in the JSONs (tumor_shape,
tumor_matrix_mri) are treated as multi-select; the rest are single-select.

Stdlib + pandas + PyYAML, matching the other two servers so it deploys the same.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import mimetypes
import sys
import urllib.parse
from collections import Counter, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "preprocess"))
from config import load_feature_config  # noqa: E402

# ---------------------------------------------------------------------------
# Module state, populated in main().
# ---------------------------------------------------------------------------
# case_id -> feature_name -> list[row-dict]; rows carry the resolved on-disk
# image path plus the overlay variant when one exists next to it.
DATA: Dict[str, Dict[str, List[dict]]] = {}
IMAGE_WHITELIST: set = set()
# feature_name -> (options, multi_select)
VOCAB: Dict[str, Tuple[List[str], bool]] = {}
# Latest verdict per target, for rendering. ("image", image_path) or
# ("subject", case_id, feature) -> annotation dict. Append-only on disk.
VERDICTS: Dict[tuple, dict] = {}
STORE: "AnnotationStore"
ANNOTATOR = ""

VERDICT_CHOICES = ("agree", "disagree", "unusable")

METADATA_COLS = (
    "case_id", "feature_name", "modality", "plane", "slice_index",
    "image_path", "mask_path", "crop_bbox", "margin_used", "ground_truth_label",
)

ANNOTATION_FIELDS = [
    "timestamp", "annotator", "level", "case_id", "feature_name",
    "modality", "plane", "slice_index", "image_path",
    "current_label", "verdict", "corrected_label", "note",
]


def esc(s) -> str:
    return html.escape("" if s is None else str(s))


def has_gt(label: str) -> bool:
    return bool(label) and label.strip().lower() not in {"unknown", "nan", "none"}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def resolve_image(recorded: str, images_root: Optional[Path]) -> Optional[Path]:
    """Map a metadata.csv path onto this machine.

    The pipeline writes container-absolute paths (/output/overlay_128/<case>/
    <feature>/<file>.png) which are not where the PNGs sit once copied out. Try
    the recorded path first, then progressively shorter suffixes joined onto
    --images-root, so a root pointing at either the parent of the case dirs or
    at a single case dir both work. None when nothing resolves.
    """
    p = Path(recorded)
    if p.is_file():
        return p
    if images_root is None:
        return None
    parts = p.parts
    for n in (3, 2, 1):  # <case>/<feature>/<file>, <feature>/<file>, <file>
        if len(parts) >= n:
            cand = images_root.joinpath(*parts[-n:])
            if cand.is_file():
                return cand
    return None


def overlay_of(image: Path) -> Optional[Path]:
    """The segmentation-overlay twin written next to the plain crop.

    metadata.csv only records the plain crop, but the pipeline also emits
    <stem>_overlay.png. The contour matters for judging soft-tissue invasion, so
    surface it as a toggle rather than leaving the annotator to guess the border.
    """
    cand = image.with_name(f"{image.stem}_overlay{image.suffix}")
    return cand if cand.is_file() else None


# ---------------------------------------------------------------------------
# Label vocabulary
# ---------------------------------------------------------------------------
def build_vocab(feature_config: Path, labels_dir: Path) -> Dict[str, Tuple[List[str], bool]]:
    """feature -> (sorted options, is_multi_select), read off the assessment JSONs.

    `assessment_key` in the feature config names the key under "imaging_features";
    the options are every value observed under it. A key whose value is a list in
    any subject is multi-select.
    """
    specs = load_feature_config(feature_config)
    key_of = {s.name: (s.assessment_key or s.name) for s in specs}

    seen: Dict[str, Counter] = defaultdict(Counter)
    is_multi: Dict[str, bool] = defaultdict(bool)
    files = sorted(labels_dir.glob("*.json"))
    for f in files:
        try:
            feats = (json.loads(f.read_text()) or {}).get("imaging_features") or {}
        except json.JSONDecodeError:
            print(f"  ! skipping unreadable {f.name}")
            continue
        for key, val in feats.items():
            if isinstance(val, list):
                is_multi[key] = True
                vals = val
            else:
                vals = [val]
            for v in vals:
                if v is not None and str(v).strip():
                    seen[key][str(v)] += 1

    vocab: Dict[str, Tuple[List[str], bool]] = {}
    for feature, key in key_of.items():
        opts = sorted(seen.get(key, {}))
        if not opts:
            print(f"  ! no values for feature '{feature}' (assessment key "
                  f"'{key}') in {len(files)} label file(s) — free-text only")
        vocab[feature] = (opts, is_multi.get(key, False))
    return vocab


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load(metadata_csv: Path, images_root: Optional[Path]) -> None:
    df = pd.read_csv(metadata_csv, dtype=str).fillna("")
    missing = [c for c in ("case_id", "feature_name", "image_path") if c not in df.columns]
    if missing:
        raise SystemExit(
            f"{metadata_csv} is missing {missing}. This UI expects the preprocess "
            f"pipeline's metadata.csv (see preprocess/outputs.py METADATA_FIELDS), "
            f"not a model results CSV — for those use medgemma_pilot/review_server.py."
        )

    n_unresolved = 0
    for rec in df.to_dict("records"):
        row = {c: rec.get(c, "") for c in METADATA_COLS}
        resolved = resolve_image(row["image_path"], images_root)
        if resolved is None:
            n_unresolved += 1
            continue
        ov = overlay_of(resolved)
        row["resolved_path"] = str(resolved)
        row["overlay_path"] = str(ov) if ov else ""
        IMAGE_WHITELIST.add(str(resolved))
        if ov:
            IMAGE_WHITELIST.add(str(ov))
        DATA.setdefault(row["case_id"], {}).setdefault(row["feature_name"], []).append(row)

    for feats in DATA.values():
        for rows in feats.values():
            rows.sort(key=lambda r: (r["modality"], r["plane"], r["slice_index"]))

    if n_unresolved:
        print(f"  ! {n_unresolved} image(s) in the CSV not found on disk — "
              f"check --images-root")


def all_rows() -> List[dict]:
    return [r for feats in DATA.values() for rows in feats.values() for r in rows]


def subject_label(case_id: str, feature: str) -> str:
    """The subject-level label for a feature: metadata.csv stamps the same
    ground truth on every crop of a (subject, feature), so any row carries it."""
    rows = DATA.get(case_id, {}).get(feature, [])
    for r in rows:
        if has_gt(r["ground_truth_label"]):
            return r["ground_truth_label"]
    return "unknown"


# ---------------------------------------------------------------------------
# Annotation store
# ---------------------------------------------------------------------------
class AnnotationStore:
    """Append-only CSV of verdicts; last row wins for display.

    Append-only (never rewritten in place) so a mid-session crash cannot lose
    earlier verdicts and so a changed mind stays auditable as its own row.
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.path.exists() or self.path.stat().st_size == 0
        self._fh = open(self.path, "a", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=ANNOTATION_FIELDS)
        if is_new:
            self._writer.writeheader()
            self._fh.flush()

    def append(self, rec: dict) -> None:
        self._writer.writerow({k: rec.get(k, "") for k in ANNOTATION_FIELDS})
        self._fh.flush()
        VERDICTS[target_key(rec)] = rec

    def replay(self) -> int:
        """Reload prior verdicts so a restarted session shows what's already done."""
        if not self.path.exists() or self.path.stat().st_size == 0:
            return 0
        n = 0
        with open(self.path, newline="") as fh:
            for rec in csv.DictReader(fh):
                VERDICTS[target_key(rec)] = rec
                n += 1
        return n


def target_key(rec: dict) -> tuple:
    if rec.get("level") == "subject":
        return ("subject", rec.get("case_id", ""), rec.get("feature_name", ""))
    return ("image", rec.get("image_path", ""))


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
PAGE = """<!doctype html><meta charset="utf-8"><title>{title}</title>
<style>
 body{{font:14px/1.45 -apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#222}}
 a{{color:#1a5fb4}} h1{{font-size:20px}} h2{{font-size:16px;margin:22px 0 8px}}
 table{{border-collapse:collapse;margin:8px 0}}
 td,th{{border:1px solid #ddd;padding:4px 10px;text-align:left}}
 .grid{{display:flex;flex-wrap:wrap;gap:14px}}
 .card{{border:1px solid #ddd;border-radius:6px;padding:8px;width:280px;background:#fff}}
 .card img{{width:264px;height:264px;object-fit:contain;background:#000;border-radius:4px}}
 .meta{{color:#666;font-size:12px}} .path{{font-family:ui-monospace,monospace;font-size:11px}}
 .summary{{background:#f7f7f9;border:1px solid #e2e2e6;border-radius:6px;padding:12px;margin-bottom:16px}}
 .badge{{display:inline-block;padding:1px 7px;border-radius:9px;font-size:12px}}
 .ok{{background:#d6f5dd;color:#0b6b2b}} .fail{{background:#fbdcdc;color:#8c1a1a}}
 .neutral{{background:#eee;color:#555}} .todo{{background:#fff2cc;color:#7a5c00}}
 .filters{{margin:10px 0}} .filters select,.filters button{{margin-right:8px;padding:3px}}
 form.ann{{margin-top:6px;border-top:1px dashed #ddd;padding-top:6px}}
 form.ann select,form.ann input{{width:100%;margin:3px 0;padding:3px;box-sizing:border-box}}
 form.ann button{{padding:4px 10px;cursor:pointer}}
 .saved{{color:#0b6b2b;font-size:12px;margin-left:6px}}
</style>
<script>
function toggleOverlay(on){{
  document.querySelectorAll('img[data-plain]').forEach(function(img){{
    var t = on && img.dataset.overlay ? img.dataset.overlay : img.dataset.plain;
    if (img.getAttribute('src') !== t) img.setAttribute('src', t);
  }});
  try {{ localStorage.setItem('overlayOn', on ? '1' : '0'); }} catch (e) {{}}
}}
function initOverlay(){{
  var box = document.getElementById('ovbox');
  if (!box) return;
  var on = false;
  try {{ on = localStorage.getItem('overlayOn') === '1'; }} catch (e) {{}}
  box.checked = on; toggleOverlay(on);
}}
async function submitAnn(ev, form){{
  ev.preventDefault();
  var fd = new FormData(form);
  var sel = form.querySelector('select[name=corrected_label][multiple]');
  if (sel) {{
    fd.delete('corrected_label');
    // ';' with no space -- the same join metadata.csv uses for multi-valued
    // ground truth ("Lobulated;Exophytic"), so the two compare directly.
    fd.set('corrected_label', Array.from(sel.selectedOptions).map(function(o){{return o.value}}).join(';'));
  }}
  var out = form.querySelector('.saved');
  out.textContent = 'saving...';
  var res = await fetch('/annotate', {{method:'POST', body:new URLSearchParams(fd)}});
  out.textContent = res.ok ? ('saved ' + new Date().toLocaleTimeString()) : 'FAILED';
  out.style.color = res.ok ? '#0b6b2b' : '#8c1a1a';
}}
window.addEventListener('DOMContentLoaded', initOverlay);
</script>
<body onload="initOverlay()">{body}</body>
"""


def img_url(path: str) -> str:
    return "/img?path=" + urllib.parse.quote(str(path))


def img_tag(row: dict) -> str:
    plain, ov = img_url(row["resolved_path"]), (
        img_url(row["overlay_path"]) if row["overlay_path"] else "")
    return (f'<img loading="lazy" src="{plain}" data-plain="{plain}"'
            f'{f" data-overlay={chr(34)}{ov}{chr(34)}" if ov else ""}>')


def overlay_toggle() -> str:
    return ('<label><input type="checkbox" id="ovbox" '
            'onchange="toggleOverlay(this.checked)"> show segmentation overlay</label>')


def verdict_badge(key: tuple) -> str:
    rec = VERDICTS.get(key)
    if not rec:
        return '<span class="badge todo">not reviewed</span>'
    v = rec.get("verdict", "")
    cls = {"agree": "ok", "disagree": "fail"}.get(v, "neutral")
    extra = f' &rarr; {esc(rec.get("corrected_label"))}' if rec.get("corrected_label") else ""
    return f'<span class="badge {cls}">{esc(v)}{extra}</span>'


def label_field(feature: str, current: str) -> str:
    """The corrected-label control: a select over the feature's vocabulary
    (multiple when the assessment key holds a list), or free text when the
    vocabulary came up empty. Only read when verdict = disagree."""
    opts, multi = VOCAB.get(feature, ([], False))
    if not opts:
        return ('<input type="text" name="corrected_label" '
                'placeholder="corrected label (if disagree)">')
    chosen = {c.strip() for c in current.split(";")} if multi else {current}
    body = "".join(
        f'<option value="{esc(o)}"{" selected" if o in chosen else ""}>{esc(o)}</option>'
        for o in opts)
    m = " multiple size=4" if multi else ""
    blank = "" if multi else '<option value="">— corrected label (if disagree) —</option>'
    return f'<select name="corrected_label"{m}>{blank}{body}</select>'


def ann_form(level: str, row_or_ids: dict, feature: str, current: str, key: tuple) -> str:
    r = row_or_ids
    hidden = "".join(
        f'<input type="hidden" name="{k}" value="{esc(r.get(k, ""))}">'
        for k in ("case_id", "modality", "plane", "slice_index", "image_path"))
    prior = VERDICTS.get(key, {})
    v_opts = "".join(
        f'<option value="{v}"{" selected" if prior.get("verdict") == v else ""}>{v}</option>'
        for v in VERDICT_CHOICES)
    return (
        f'<form class="ann" onsubmit="submitAnn(event,this)">'
        f'<input type="hidden" name="level" value="{level}">'
        f'<input type="hidden" name="feature_name" value="{esc(feature)}">'
        f'<input type="hidden" name="current_label" value="{esc(current)}">'
        f"{hidden}"
        f'<select name="verdict">{v_opts}</select>'
        f"{label_field(feature, current)}"
        f'<input type="text" name="note" placeholder="note (optional)" '
        f'value="{esc(prior.get("note", ""))}">'
        f'<button type="submit">save</button><span class="saved"></span>'
        f"</form>")


def progress(keys: List[tuple]) -> Tuple[int, int]:
    return sum(1 for k in keys if k in VERDICTS), len(keys)


def index_html() -> str:
    rows = []
    for case_id in sorted(DATA):
        feats = DATA[case_id]
        keys = [("image", r["image_path"]) for rows_ in feats.values() for r in rows_]
        skeys = [("subject", case_id, f) for f in feats]
        di, ti = progress(keys)
        ds, ts = progress(skeys)
        rows.append(
            f'<tr><td><a href="/subject?id={urllib.parse.quote(case_id)}">{esc(case_id)}</a></td>'
            f"<td>{len(feats)}</td><td>{ti}</td>"
            f"<td>{di}/{ti}</td><td>{ds}/{ts}</td></tr>")

    features = sorted({r["feature_name"] for r in all_rows()})
    modalities = sorted({r["modality"] for r in all_rows() if r["modality"]})
    planes = sorted({r["plane"] for r in all_rows() if r["plane"]})

    def sel(name, values):
        opts = "".join(f'<option value="{esc(v)}">{esc(v)}</option>' for v in values)
        return f'<select name="{name}"><option value="">all {name}s</option>{opts}</select>'

    di, ti = progress([("image", r["image_path"]) for r in all_rows()])
    body = (
        '<div class="summary">'
        f"<h1>Label review — {len(DATA)} subject(s), {ti} image(s)</h1>"
        f'<div>annotator: <b>{esc(ANNOTATOR or "(unset — pass --annotator)")}</b>'
        f' &nbsp;·&nbsp; image-level progress: <b>{di}/{ti}</b>'
        f' &nbsp;·&nbsp; writing to <code>{esc(STORE.path)}</code></div>'
        '<form class="filters" action="/browse" method="get">'
        + sel("feature", features) + sel("modality", modalities) + sel("plane", planes)
        + '<label><input type="checkbox" name="todo" value="1"> only not-reviewed</label> '
        '<button type="submit">browse across subjects</button></form>'
        "</div>"
        "<table><tr><th>subject</th><th>features</th><th>images</th>"
        "<th>images reviewed</th><th>subject-level reviewed</th></tr>"
        + "".join(rows) + "</table>")
    return PAGE.format(title="Label review", body=body)


def subject_html(case_id: str) -> str:
    feats = DATA[case_id]
    out = [f'<h1><a href="/">&larr; all subjects</a> &nbsp; {esc(case_id)}</h1>',
           f'<div class="summary">{overlay_toggle()}</div>']
    for feature in sorted(feats):
        rows = feats[feature]
        cur = subject_label(case_id, feature)
        skey = ("subject", case_id, feature)
        out.append(
            f"<h2>{esc(feature)} <span class='meta'>current label:</span> "
            f"<b>{esc(cur)}</b> &nbsp; {verdict_badge(skey)}</h2>"
            '<div class="card" style="width:420px">'
            "<div class='meta'>subject-level verdict — does this label hold for the "
            "lesion overall, across every image below?</div>"
            + ann_form("subject", {"case_id": case_id}, feature, cur, skey)
            + "</div>")
        cards = []
        for r in rows:
            key = ("image", r["image_path"])
            ov = "" if r["overlay_path"] else " · <span class='meta'>no overlay</span>"
            cards.append(
                '<div class="card">' + img_tag(r) +
                f'<div class="meta">{esc(r["modality"])} · {esc(r["plane"])} · '
                f'slice {esc(r["slice_index"])}{ov}</div>'
                f'<div class="meta path" title="{esc(r["resolved_path"])}">'
                f'{esc(Path(r["resolved_path"]).name)}</div>'
                f'<div>label: <b>{esc(r["ground_truth_label"] or "unknown")}</b> '
                f"{verdict_badge(key)}</div>"
                + ann_form("image", r, feature, r["ground_truth_label"], key)
                + "</div>")
        out.append('<div class="grid">' + "".join(cards) + "</div>")
    return PAGE.format(title=f"{case_id} — review", body="".join(out))


def browse_html(feature: str, modality: str, plane: str, todo_only: bool) -> str:
    rows = [r for r in all_rows()
            if (not feature or r["feature_name"] == feature)
            and (not modality or r["modality"] == modality)
            and (not plane or r["plane"] == plane)]
    if todo_only:
        rows = [r for r in rows if ("image", r["image_path"]) not in VERDICTS]
    rows.sort(key=lambda r: (r["case_id"], r["feature_name"], r["modality"], r["plane"]))

    by_label = Counter(r["ground_truth_label"] or "unknown" for r in rows)
    crit = " · ".join(x for x in (f"feature={feature}" if feature else "",
                                  f"modality={modality}" if modality else "",
                                  f"plane={plane}" if plane else "",
                                  "not-reviewed only" if todo_only else "") if x)
    dist = ", ".join(f"{esc(k)}: {v}" for k, v in sorted(by_label.items()))
    cards = []
    for r in rows:
        key = ("image", r["image_path"])
        cards.append(
            '<div class="card">' + img_tag(r) +
            f'<div class="meta"><a href="/subject?id={urllib.parse.quote(r["case_id"])}">'
            f'{esc(r["case_id"])}</a> · {esc(r["feature_name"])}</div>'
            f'<div class="meta">{esc(r["modality"])} · {esc(r["plane"])} · '
            f'slice {esc(r["slice_index"])}</div>'
            f'<div>label: <b>{esc(r["ground_truth_label"] or "unknown")}</b> '
            f"{verdict_badge(key)}</div>"
            + ann_form("image", r, r["feature_name"], r["ground_truth_label"], key)
            + "</div>")
    body = (f'<h1><a href="/">&larr; all subjects</a> &nbsp; {len(rows)} image(s)</h1>'
            f'<div class="summary"><div class="meta">{esc(crit) or "no filter"}</div>'
            f"<div>current labels — {dist or '—'}</div>{overlay_toggle()}</div>"
            '<div class="grid">' + "".join(cards) + "</div>")
    return PAGE.format(title="browse", body=body)


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
        one = lambda k: (qs.get(k) or [""])[0]  # noqa: E731

        if parsed.path == "/":
            self._send(index_html().encode(), "text/html; charset=utf-8")
        elif parsed.path == "/subject":
            cid = one("id")
            if cid in DATA:
                self._send(subject_html(cid).encode(), "text/html; charset=utf-8")
            else:
                self._send(b"unknown subject", "text/plain", 404)
        elif parsed.path == "/browse":
            page = browse_html(one("feature"), one("modality"), one("plane"),
                               one("todo") == "1")
            self._send(page.encode(), "text/html; charset=utf-8")
        elif parsed.path == "/img":
            path = one("path")
            if path not in IMAGE_WHITELIST:   # only serve images named in the CSV
                self._send(b"forbidden", "text/plain", 403)
                return
            p = Path(path)
            if not p.is_file():
                self._send(b"image not found on disk", "text/plain", 404)
                return
            self._send(p.read_bytes(),
                       mimetypes.guess_type(str(p))[0] or "application/octet-stream")
        else:
            self._send(b"not found", "text/plain", 404)

    def do_POST(self) -> None:
        if urllib.parse.urlparse(self.path).path != "/annotate":
            self._send(b"not found", "text/plain", 404)
            return
        n = int(self.headers.get("Content-Length") or 0)
        form = urllib.parse.parse_qs(self.rfile.read(n).decode())
        get = lambda k: (form.get(k) or [""])[0].strip()  # noqa: E731

        level, verdict = get("level"), get("verdict")
        if level not in {"image", "subject"} or verdict not in VERDICT_CHOICES:
            self._send(b"bad level/verdict", "text/plain", 400)
            return
        # A corrected label is only meaningful when the annotator disagrees;
        # drop a stale select value on agree/unusable so the CSV stays honest.
        corrected = get("corrected_label") if verdict == "disagree" else ""

        rec = {
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "annotator": ANNOTATOR,
            "level": level,
            "case_id": get("case_id"),
            "feature_name": get("feature_name"),
            "modality": get("modality") if level == "image" else "",
            "plane": get("plane") if level == "image" else "",
            "slice_index": get("slice_index") if level == "image" else "",
            "image_path": get("image_path") if level == "image" else "",
            "current_label": get("current_label"),
            "verdict": verdict,
            "corrected_label": corrected,
            "note": get("note"),
        }
        STORE.append(rec)
        self._send(json.dumps({"ok": True}).encode(), "application/json")

    def log_message(self, *a) -> None:  # quiet: one line per image request is noise
        pass


def main() -> None:
    global STORE, ANNOTATOR, VOCAB

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--metadata", type=Path, required=True,
                    help="metadata.csv written by the preprocess pipeline")
    ap.add_argument("--images-root", type=Path, default=None,
                    help="directory the PNGs actually live in, when the CSV's "
                         "recorded paths are container-absolute")
    ap.add_argument("--feature-config", type=Path,
                    default=Path(__file__).resolve().parent.parent / "preprocess" / "feature_config.yaml",
                    help="feature config; its assessment_key names the label key")
    ap.add_argument("--labels-dir", type=Path, required=True,
                    help="assessment JSONs — the corrected-label vocabulary comes "
                         "from the values observed here")
    ap.add_argument("--out", type=Path, default=Path("annotations.csv"),
                    help="append-only CSV of verdicts (resumed on restart)")
    ap.add_argument("--annotator", default="", help="recorded on every verdict")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    # A server's startup lines are the only feedback before it blocks forever;
    # line-buffer so they show up immediately when stdout is a log file.
    sys.stdout.reconfigure(line_buffering=True)

    ANNOTATOR = args.annotator
    print(f"vocabulary from {args.labels_dir}")
    VOCAB = build_vocab(args.feature_config, args.labels_dir)
    for feat, (opts, multi) in sorted(VOCAB.items()):
        print(f"  {feat}: {len(opts)} option(s){' (multi-select)' if multi else ''}"
              f"{' — ' + ', '.join(opts) if opts else ''}")

    load(args.metadata, args.images_root)
    n_img = sum(len(v) for feats in DATA.values() for v in feats.values())
    print(f"loaded {len(DATA)} subject(s), {n_img} image(s) from {args.metadata}")

    STORE = AnnotationStore(args.out)
    n_prior = STORE.replay()
    if n_prior:
        print(f"resumed {n_prior} prior verdict(s) from {args.out}")
    if not ANNOTATOR:
        print("  ! no --annotator given; verdicts will be written with a blank name")

    print(f"serving at http://{args.host}:{args.port}  (Ctrl-C to stop)")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
