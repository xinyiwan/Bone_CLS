"""Build BTI-RADS 2.0 (MRI + Medium) inputs for the selected subset from the merged
clean dataset, with fixed rules (no LLM). One row per study (info_key).
Empty = not available -> field is left out when querying the calculator.

Sizes the LLM missed are listed in <out>.size_missing.csv (written only when no
--size-manual is given). Copy it, fill size_mm by hand and pass it with --size-manual.
A non-numeric entry (NaN, unknown, ...) means "exclude this study" from the output.

  python build_bti_input.py [--size-manual size_manual.csv]
"""
import argparse, ast, json, re
from pathlib import Path
import pandas as pd

R = Path("/Users/xinyi/Documents/github/Bone_CLS")
ap = argparse.ArgumentParser()
ap.add_argument("--merged", default=R / "output/merged_data/kira-1104-combined_dataset.filtered.csv")
ap.add_argument("--subset", default=R / "output/subset/subset_waterfill_filtered1106-15pct.csv")
ap.add_argument("--review", default="/Users/xinyi/Documents/github/BTRecordsLLM/output/subset_pct15/subset_review.json",
                help="manual review of LLM2 fields; verdict 'incorrect' overrides the LLM value (empty string to skip)")
ap.add_argument("--size-manual", default=None)
ap.add_argument("--clarify", action="store_true",
                help="treat 'not mentioned in the report' as 'absent' (0) for: fibrous matrix, fluid without "
                     "fluid-fluid levels, soft tissue invasion, absence of contrast enhancement (=1)")
ap.add_argument("-o", "--out", default=R / "output/BTI/input_data/bti_input_subset_clarify.csv")
a = ap.parse_args()

# studies left out on purpose: info_key -> reason
EXCLUDE = {"3773002": "BONE_AI_172: our images match study 5032373 (corrected there), not this one"}
YN = {"yes": 1, "no": 0}
LOC = {  # BTI field -> regex on 'Skeletal Location' (LLM1), lower-case
    "Bone type_flat bone": r"ili|ischi|pubi|pelvi|scapul|sternum|\brib|skull|cranial|calvari",
    "Bone type_hand/foot": r"metacarp|metatars|phalan|carpal|tarsal|calcane|talus|\bhand|\bfoot|finger|\btoe",
    "Anatomical region_axial (spine/pelvis)": r"spine|vertebra|lumbar|thoracic|cervical|sacr|coccyx|atlas|\b[lstc]\d{1,2}\b|ili|ischi|pubi|pelvi|acetab",
    "Anatomical region_body walls": r"\brib|sternum|chest wall|abdominal wall|thoracic wall",
}
SHAPE = {"round/oval": "Shape_round/oval", "irregular": "Shape_irregular", "lobulated": "Shape_lobulated"}


# info_key -> {LLM2 field: corrected value}, only where the reviewer marked it 'incorrect';
# studies without a review keep the LLM values
FIX = {}
if a.review:
    for v in json.load(open(a.review)).values():
        fx = {f: x["corrected_value"] for f, x in v["fields"].items()
              if x["verdict"] == "incorrect" and x["corrected_value"] is not None}
        if fx:
            FIX[str(v["info_key"])] = fx
    print(f"review overrides: {sum(map(len, FIX.values()))} fields in {len(FIX)} studies")


def yn(v, clarify=False):  # clarify: anything not stated Yes -> 0 (the report would have mentioned it)
    return YN.get(str(v).strip().lower(), 0 if clarify else "")


def rules(r):
    c, i = ast.literal_eval(r.clinical_data), ast.literal_eval(r.imaging_features)
    i.update(FIX.get(r.info_key, {}))
    o = {"case_id": r.info_key, "subject_code": r.subject_code}
    born, seen = pd.to_datetime(r.fechaNaci, dayfirst=True), pd.to_datetime(r.fechaHoraRealizacion).tz_localize(None)
    o["Age"] = (seen - born).days // 365.25 if pd.notna(born) else ""
    o["Size (mm)"] = pd.to_numeric(i.get("Max Diameter (longest axis, mm)"), errors="coerce")
    o["Neoplasm history_presence"] = yn(c.get("History of neoplasm"))
    o["Suspicion for metastases_presence"] = yn(c.get("Suspected metastatic disease"))
    p = str(c.get("Progression")).strip().lower()
    o["Expandability_presence"] = "NA" if p == "not applicable" else yn(p)
    loc = str(c.get("Skeletal Location") or "").strip().lower()
    for k, rx in LOC.items():  # no usable location -> leave empty rather than guess 0
        o[k] = "" if loc in ("", "not specified") else int(bool(re.search(rx, loc)))
    o["Transversal location_subperiosteal"] = ""  # not available
    sh = str(i.get("Shape")).strip().lower()
    for k, f in SHAPE.items():  # 'others' / not specified -> all empty
        o[f] = "" if sh not in SHAPE else int(k == sh)
    o["MRI matrix_fibrous"] = yn(i.get("Fibrous Component"), a.clarify)
    o["MRI matrix_liquid without liquid-liquid levels"] = yn(i.get("Fluid/Hemorrhage"), a.clarify)
    ce = str(c.get("MRI contrast enhancement")).strip().lower()  # 'absence/unknown' is ambiguous -> empty
    o["MRI contrast enhancement_absence"] = (
        1 if ce == "no enhancement" else 0 if "enhancement" in ce else 1 if a.clarify else "")
    o["MRI soft tissue invasion_presence"] = yn(i.get("Soft Tissue Invasion"), a.clarify)
    o["Do not touch_presence"] = ""  # not available
    return o


m = pd.read_csv(a.merged, dtype={"info_key": str})
m = m[m.subject_code.isin(pd.read_csv(a.subset).subject_code) & ~m.info_key.isin(EXCLUDE)]
out = pd.DataFrame([rules(r) for r in m.itertuples()])

if a.size_manual:
    # ragged rows are fine: only the first 3 columns are used
    man = pd.read_csv(a.size_manual, header=0, names=["info_key", "subject_code", "size_mm", "note", "x"],
                      dtype=str, keep_default_na=False).set_index("info_key")["size_mm"]
    num = pd.to_numeric(man, errors="coerce")
    drop = set(man.index[num.isna()])          # NaN / unknown / blank -> exclude
    out["Size (mm)"] = out["Size (mm)"].fillna(out.case_id.map(num))
    print(f"excluded by manual review (no usable size): {len(drop)} studies")
    out = out[~out.case_id.isin(drop)]

out_path = Path(a.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(out_path, index=False)
miss = out[out["Size (mm)"].isna()][["case_id", "subject_code"]].rename(columns={"case_id": "info_key"})
if not a.size_manual:
    miss.to_csv(out_path.with_suffix(".size_missing.csv"), index=False)
print(f"{len(out)} studies ({out.subject_code.nunique()} subjects) -> {out_path}")
print(f"size still missing: {len(miss)}")
print("non-empty per field:\n", out.iloc[:, 2:].replace("", pd.NA).notna().sum().to_string())
