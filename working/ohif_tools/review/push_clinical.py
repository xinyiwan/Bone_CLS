"""
Push per-subject clinical info (for the Bone-AI review panel) into Orthanc as a
STUDY ATTACHMENT (content-type 1024), read from the combined clinical CSV.

Matches each Orthanc study by (PatientID, StudyDescription) against the CSV's
(patient_id, session). Writes a small JSON the panel displays read-only:

  {
    "age": 38, "sex": "Male",
    "symptoms": ["Pain", "Hip pain"],
    "history_of_neoplasm": "No", "suspected_metastasis": "No",
    "anatomy_options": ["humerus_right", "scapula_right", "clavicula_left"]
  }

anatomy_options come from top1/top2/top3_label (blank ones dropped). Columns
skeletal_location / location_within_bone are intentionally ignored.

Only depends on the Python standard library.

USAGE
-----
    python push_clinical.py --csv /path/combined_bone_data_anatomy_cli.csv
    python push_clinical.py --csv ... --host drogo --scheme https --base /orthanc --user U --password P
"""
import csv
import ast
import json
import base64
import argparse
import urllib.request

CLINICAL_ATTACHMENT_ID = 1024


def parse_symptoms(raw):
    raw = (raw or "").strip()
    if not raw:
        return []
    try:
        val = ast.literal_eval(raw)
        if isinstance(val, (list, tuple)):
            return [str(x) for x in val if str(x).strip()]
    except (ValueError, SyntaxError):
        pass
    return [raw]


def clean(v):
    v = (v or "").strip()
    return "" if v.lower() in ("", "nan", "not specified", "none") else v


def build_clinical(row):
    try:
        age = int(float(row.get("age", "")))
    except (ValueError, TypeError):
        age = None
    sex = {"0": "Male", "1": "Female"}.get(str(row.get("gender", "")).strip(), "")
    anatomy_options = [c for c in (clean(row.get("top1_label")),
                                   clean(row.get("top2_label")),
                                   clean(row.get("top3_label"))) if c]
    return {
        "age": age,
        "sex": sex,
        "symptoms": parse_symptoms(row.get("symptoms")),
        "history_of_neoplasm": clean(row.get("history_of_neoplasm")) or "Unknown",
        "suspected_metastasis": clean(row.get("suspected_metastasis")) or "Unknown",
        "anatomy_options": anatomy_options,
    }


def main():
    ap = argparse.ArgumentParser(description="Push clinical info into Orthanc study attachments.")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8042)
    ap.add_argument("--scheme", choices=["http", "https"], default="http")
    ap.add_argument("--base", default="")
    ap.add_argument("--user", default=None)
    ap.add_argument("--password", default=None)
    args = ap.parse_args()

    root = f"{args.scheme}://{args.host}:{args.port}{args.base}"
    headers = {}
    if args.user is not None:
        token = base64.b64encode(f"{args.user}:{args.password or ''}".encode()).decode()
        headers["Authorization"] = f"Basic {token}"

    def req(method, path, body=None, ctype=None):
        h = dict(headers)
        if ctype:
            h["Content-Type"] = ctype
        r = urllib.request.Request(root + path, data=body, headers=h, method=method)
        with urllib.request.urlopen(r, timeout=30) as resp:
            return resp.read()

    # index CSV by (patient_id, session)
    index = {}
    with open(args.csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            index[(row["patient_id"].strip(), row["session"].strip())] = row

    study_ids = json.loads(req("GET", "/studies"))
    pushed = missing = 0
    for sid in study_ids:
        study = json.loads(req("GET", f"/studies/{sid}"))
        pid = study.get("PatientMainDicomTags", {}).get("PatientID", "").strip()
        desc = study.get("MainDicomTags", {}).get("StudyDescription", "").strip()
        row = index.get((pid, desc))
        if not row:
            print(f"  ? {pid} / {desc}: no CSV match")
            missing += 1
            continue
        clinical = build_clinical(row)
        req("PUT", f"/studies/{sid}/attachments/{CLINICAL_ATTACHMENT_ID}",
            body=json.dumps(clinical).encode(), ctype="application/json")
        pushed += 1
        print(f"  {pid} / {desc}: age={clinical['age']} sex={clinical['sex']} "
              f"symptoms={clinical['symptoms']} anatomy={clinical['anatomy_options']}")

    print(f"\nPushed clinical to {pushed} study(ies); {missing} without a CSV match.")


if __name__ == "__main__":
    main()
