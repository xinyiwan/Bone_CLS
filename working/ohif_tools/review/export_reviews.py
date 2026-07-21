"""
Export Bone-AI review forms (assessment JSON) stored as Orthanc STUDY ATTACHMENTS.

Each reviewed study carries the review under attachment content-type 1025
(see PanelBoneAI "Save Review"). This walks every study, pulls that attachment,
and writes:
  <output>/<PatientID>__<StudyDescription>.json   (one per subject-session)
  <output>/all_reviews.json                        (combined, keyed by PatientID)

Only depends on the Python standard library.

USAGE
-----
    python export_reviews.py                          # local Orthanc :8042
    python export_reviews.py --host drogo --port 8042
    python export_reviews.py --scheme https --base /orthanc --user U --password P
"""
import os
import json
import base64
import argparse
import urllib.request

ATTACHMENT_ID = 1025


def main():
    ap = argparse.ArgumentParser(description="Export review attachments from Orthanc.")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8042)
    ap.add_argument("--scheme", choices=["http", "https"], default="http")
    ap.add_argument("--base", default="", help="base path prefix, e.g. /orthanc")
    ap.add_argument("--user", default=None)
    ap.add_argument("--password", default=None)
    ap.add_argument("--output", default="./reviews_out")
    args = ap.parse_args()

    root = f"{args.scheme}://{args.host}:{args.port}{args.base}"
    headers = {}
    if args.user is not None:
        token = base64.b64encode(f"{args.user}:{args.password or ''}".encode()).decode()
        headers["Authorization"] = f"Basic {token}"

    def get(path, raw=False):
        req = urllib.request.Request(root + path, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        return data if raw else json.loads(data)

    os.makedirs(args.output, exist_ok=True)
    combined = {}
    n = 0
    for sid in get("/studies"):
        study = get(f"/studies/{sid}")
        pid = study.get("PatientMainDicomTags", {}).get("PatientID", sid)
        desc = study.get("MainDicomTags", {}).get("StudyDescription", "")
        attachments = get(f"/studies/{sid}/attachments")
        if str(ATTACHMENT_ID) not in [str(a) for a in attachments]:
            continue
        try:
            review = json.loads(get(f"/studies/{sid}/attachments/{ATTACHMENT_ID}/data", raw=True))
        except Exception as e:
            print(f"  ! {pid}: could not read attachment ({e})")
            continue
        fname = f"{pid}__{desc}.json".replace("/", "_")
        with open(os.path.join(args.output, fname), "w") as f:
            json.dump(review, f, indent=2)
        combined[pid] = {
            "studyInstanceUID": study.get("MainDicomTags", {}).get("StudyInstanceUID"),
            "studyDescription": desc,
            "review": review,
        }
        n += 1
        print(f"  {pid}  ({desc})")

    with open(os.path.join(args.output, "all_reviews.json"), "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nExported {n} review(s) to {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
