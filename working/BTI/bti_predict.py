"""Query the BTI-RADS 2.0 calculator (MRI + Medium) for each row of a CSV.

CSV columns = form field names (see FIELDS). Only Age and "Size (mm)" are mandatory;
empty cells are omitted (the site predicts from partial data). Binary fields: 1/0,
"Expandability_presence" also accepts NA.
Saves per row in OUT/<id>/: result.json (score, probability, request) and shap.png;
OUT/explanation.json holds the static BTI-RADS legend table (shared by all cases).
Existing result.json is skipped, so re-runs never re-query the server.

Versions: --prev <old version dir> copies cases whose request is unchanged and queries only
new/changed ones; OUT/changes.csv records copied / rerun / new / dropped per case.

  python bti_predict.py in.csv out_dir [--prev old_dir] [--id-col case_id] [--delay 10] [--dry-run]
"""
import argparse, csv, base64, json, shutil, time, urllib.parse, urllib.request, uuid
from pathlib import Path

BASE = "https://bti-rads.cic-it-nancy.fr"
IMAGING, FEATURES = "mr", "medium"
FIELDS = [  # all inputs enabled for MRI / Medium
    "Age", "Size (mm)", "Neoplasm history_presence", "Suspicion for metastases_presence",
    "Expandability_presence", "Bone type_flat bone", "Bone type_hand/foot",
    "Anatomical region_axial (spine/pelvis)", "Anatomical region_body walls",
    "Transversal location_subperiosteal", "Shape_irregular", "Shape_lobulated",
    "Shape_round/oval", "MRI matrix_fibrous", "MRI matrix_liquid without liquid-liquid levels",
    "MRI contrast enhancement_absence", "MRI soft tissue invasion_presence",
    "Do not touch_presence",
]


def build_request(row):
    d = {k: str(row[k]).strip() for k in FIELDS if str(row.get(k, "")).strip() not in ("", "nan")}
    for k in ("Age", "Size (mm)"):
        if k not in d:
            raise ValueError(f"{k} is mandatory")
    for k in ("Age", "Size (mm)"):  # "38.0" -> "38", as typed in the form
        d[k] = f"{float(d[k]):g}"
    # derived fields, mirrors btiSave() in the site's btiEdit.js
    if "Expandability_presence" in d:
        p = d["Expandability_presence"]
        na = p.upper() == "NA"
        d["Expandability_absence"] = 0 if na else 1 - int(p)
        d["Expandability_not applicable"] = int(na)
        d["Expandability_presence"] = 0 if na else int(p)
    if "MRI soft tissue invasion_presence" in d:
        d["MRI soft tissue invasion_absence"] = 1 - int(d["MRI soft tissue invasion_presence"])
    return d


def _call(method, path, data=None):
    body = urllib.parse.urlencode(data).encode() if data else None
    req = urllib.request.Request(BASE + path, data=body, method=method)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def predict(request, wait=5, max_polls=60):
    r = _call("PUT", "/bti/api/btis", {
        "bti_id": str(uuid.uuid4()), "bti_imaging": IMAGING, "bti_features": FEATURES,
        "bti_request": json.dumps(request)})
    if r.get("status") != "ok":
        raise RuntimeError(f"server: {r}")
    time.sleep(wait)
    for _ in range(max_polls):
        j = _call("GET", "/bti/api/btis/" + r["uid"])
        if j.get("bti_result") is not None:
            # the site returns bare NaN tokens (not valid JSON); parse_constant leaves the base64
            # image string untouched (a plain str.replace("NaN", ...) corrupts it when it contains "NaN")
            return r["uid"], json.loads(j["bti_result"], parse_constant=lambda c: None)
        time.sleep(2)
    raise TimeoutError(r["uid"])


def save(out, uid, request, res):
    out.mkdir(parents=True, exist_ok=True)
    pred = res["Prediction"]
    (out / "result.json").write_text(json.dumps({
        "uid": uid, "request": request,
        "score": pred["BTIRADS 2.0 score"], "probability": pred["Probability of malignancy (%)"]}, indent=1))
    # the explanation table is the static BTI-RADS legend; store once at the root,
    # and keep a per-case copy only if the site ever returns something different
    shared = out.parent / "explanation.json"
    exp = json.dumps(res["Explanation"], indent=1)
    if not shared.exists():
        shared.write_text(exp)
    elif shared.read_text() != exp:
        print("  WARNING: explanation differs from shared one, saved per case")
        (out / "explanation.json").write_text(exp)
    # result.json is written first so a bad image never costs a re-query
    (out / "shap.png").write_bytes(base64.b64decode(res["Plot"]["Shap decision plot"]))
    return pred


def diff(old, new):
    return ", ".join(sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); 
    ap.add_argument("out")
    ap.add_argument("--prev", help="previous version folder: cases whose request is unchanged are copied "
                                   "from it, only new/changed cases are queried")
    ap.add_argument("--id-col", default="case_id")
    ap.add_argument("--delay", type=float, default=10, help="seconds between queries")
    ap.add_argument("--dry-run", action="store_true", help="print payloads, no server calls")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    prev = Path(a.prev) if a.prev else None
    if prev and not any(prev.glob("*/result.json")):  # e.g. pointing at analysis/ instead of prediction/
        raise SystemExit(f"--prev {prev} has no <case>/result.json; expected a prediction/<version> folder")
    log, ids, fails = [], set(), 0  # log -> out/changes.csv
    for i, row in enumerate(csv.DictReader(open(a.csv))):
        cid = row.get(a.id_col) or str(i)
        ids.add(cid)
        out = Path(a.out) / cid
        if (out / "result.json").exists():
            print(cid, "skip (done)"); continue
        try:
            req = build_request(row)
            action, why = "new" if prev else "query", ""
            if prev and (prev / cid / "result.json").exists():
                old = json.loads((prev / cid / "result.json").read_text())["request"]
                if old == req:
                    if not a.dry_run:
                        shutil.copytree(prev / cid, out)
                    log.append((cid, "copied", "")); print(cid, "copied (unchanged)"); continue
                action, why = "rerun", diff(old, req)
            if a.dry_run:
                print(cid, action, why or json.dumps(req)); continue
            uid, res = predict(req)
            p = save(out, uid, req, res)
            log.append((cid, action, why))
            print(cid, action, p["BTIRADS 2.0 score"], p["Probability of malignancy (%)"], why, flush=True)
            fails = 0
        except Exception as e:  # keep going; failed rows are retried on next run
            print(cid, "FAILED", e, flush=True)
            fails += 1
            if fails >= 3:  # server trouble: stop instead of hammering it
                raise SystemExit("3 failures in a row, stopping. Re-run later; finished cases are skipped.")
        time.sleep(a.delay)
    if prev:  # cases that were in the previous version but are no longer in the input
        log += [(p.name, "dropped", "not in input") for p in sorted(prev.iterdir()) if p.is_dir() and p.name not in ids]
    if prev and not a.dry_run:
        for f in ("explanation.json",):  # shared legend table
            if (prev / f).exists() and not (Path(a.out) / f).exists():
                shutil.copy(prev / f, Path(a.out) / f)
        with open(Path(a.out) / "changes.csv", "w", newline="") as f:  # what this version did per case
            csv.writer(f).writerows([("case_id", "action", "changed_fields")] + log)
    for r in log:
        if r[1] in ("dropped",):
            print(*r)
