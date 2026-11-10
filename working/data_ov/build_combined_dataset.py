"""Combine original report data, LLM extractions, annotations, enchondroma review
and on-disk segmentation status into one CSV (one row per info_key).

Output keeps review_server.py's inputs (subject_code, valoracion_en_clean,
valoracion_thinking, reasoning, final_output, META_COLS), so it can be passed
straight to `review_server.py -f`.
"""
import argparse
import ast
import json
import re
from pathlib import Path

import pandas as pd

R = Path("/Users/xinyi/Documents/github/Bone_CLS")
ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("--orig", default=R / "kira-0515-seg.merged.csv")
ap.add_argument("--llm1", default="/Volumes/SanDisk/BONE-AI/workspace/Bone_AI/BTRecordsLLM/output/kira-0515-llm-en-split.csv")
ap.add_argument("--llm2", default="/Users/xinyi/Documents/github/BTRecordsLLM/output/all_subjects/imaging_features_zero_DeepSeek-R1-0528-Qwen3-8B.csv")
ap.add_argument("--ench", default="/Users/xinyi/Documents/github/EnchondromaCheck/include_enchondroma.csv")
ap.add_argument("--data-root", default="/Volumes/SanDisk/BONE-AI/tmp_sorted_data")
ap.add_argument("-o", "--out", default=R / "output/merged_data/kira-1104-combined_dataset.csv")
a = ap.parse_args()

orig = pd.read_csv(a.orig)                       # base: original report + latest annotations
orig = orig.drop(columns=["Unnamed: 24", "size", "note"])
orig["If_segmented"] = orig["If_segmented"].str.strip()

llm1 = pd.read_csv(a.llm1, usecols=["info_key", "valoracion_en", "valoracion_en_clean",
                                    "extracted_data", "reasoning", "raw_output"])
llm1 = llm1.rename(columns={"extracted_data": "clinical_data", "reasoning": "clinical_reasoning"})


def answer_json(raw):
    """JSON from the model's final answer (text after the last </think>), or None.
    The LLM1 parser took the last ```json block anywhere, which for bare-JSON answers
    was the empty template inside the reasoning -> blank clinical_data."""
    tail = str(raw).rsplit("</think>", 1)[-1]
    for b in reversed(re.findall(r"```json\s*({.*?})\s*```", tail, re.S) or re.findall(r"({.*?})", tail, re.S)):
        try:
            return json.loads(b)
        except ValueError:
            pass


# repair rows where the parser kept the empty template (blank Radiology Request Reason)
# but the real answer is in raw_output
fixed = 0
for i, r in llm1.iterrows():
    cd = ast.literal_eval(r.clinical_data) if isinstance(r.clinical_data, str) else {}
    new = answer_json(r.raw_output)
    if new and not str(cd.get("Radiology Request Reason", "")).strip() and str(new.get("Radiology Request Reason", "")).strip():
        llm1.at[i, "clinical_data"] = str({**cd, **new})
        fixed += 1
print("LLM1 clinical_data repaired from raw_output:", fixed)
llm1 = llm1.drop(columns="raw_output")
llm2 = pd.read_csv(a.llm2, usecols=["info_key", "reasoning", "extracted_data", "final_output", "note"])
llm2 = llm2.rename(columns={"extracted_data": "imaging_features", "reasoning": "reasoning_imaging"})
llm2["note"] = llm2["note"].str.strip()   # "not found" = no source data on disk
llm2["reasoning"] = llm2["reasoning_imaging"]    # review_server reads 'reasoning'

df = orig.merge(llm1, on="info_key", how="left", validate="1:1").merge(llm2, on="info_key", how="left", validate="1:1")
df["study_date"] = pd.to_datetime(df["fechaHoraRealizacion"], utc=True).dt.strftime("%Y%m%d")

# --- enchondroma manual check (keyed on subject + study date) ---
e = pd.read_csv(a.ench, dtype={"study_date": str})
e = e.rename(columns={"decision": "enchondroma_decision", "study_folder": "enchondroma_study_folder",
                      "reason_too_small": "enchondroma_too_small", "reason_other": "enchondroma_reason_other"})
e = e.drop(columns="timestamp").drop_duplicates(["subject_code", "study_date"], keep="last")
df = df.merge(e, on=["subject_code", "study_date"], how="left")
print("enchondroma rows matched:", df.enchondroma_decision.notna().sum(), "of", len(e))

# --- manual label corrections (applied by info_key) ---
# manual_label_fixes.csv: hand-corrected palabra_manual / cancer (columns left empty are not touched)
# uncertain_labels.csv:   diagnosis for 'uncertain by reports' studies (palabra_manual_corrected, single Spanish term)
# original kept in palabra_manual_orig
df["palabra_manual_orig"] = df["palabra_manual"]
LC = R / "output/label_correction"
ik = df.info_key.astype(str)
for f, col_in, col_out in [("manual_label_fixes.csv", "palabra_manual", "palabra_manual"),
                           ("manual_label_fixes.csv", "cancer", "cancer"),
                           ("uncertain_labels.csv", "palabra_manual_corrected", "palabra_manual")]:
    if (LC / f).exists():
        fx = pd.read_csv(LC / f, dtype=str).dropna(subset=[col_in])
        fx = fx[fx[col_in].str.strip() != ""].set_index("info_key")[col_in].str.strip()
        hit = ik.isin(fx.index)
        df.loc[hit, col_out] = ik[hit].map(fx)
        print(f"{f}: {col_out} corrected in {hit.sum()} rows")

# --- segmentation status from disk: <subj>/<session>/segmentation_history/segs/*.nii.gz ---
# Session folder date can differ from the report date by days, so each segmented
# session is assigned to the subject's row with the nearest report date.
root = Path(a.data_root)
df["n_seg_files"] = 0
df["n_final_files"] = 0   # FINAL_*.nii.gz directly under segmentation_history/ (info only)
df["seg_session"] = ""
dates = pd.to_datetime(df["study_date"])
for h in sorted(root.glob("*/*/segmentation_history")):
    subj, sess = h.parts[-3], h.parts[-2]
    n = len([p for p in (h / "segs").glob("*.nii.gz") if not p.name.startswith("._")])
    nf = len([p for p in h.glob("*.nii.gz") if not p.name.startswith("._")])
    m = re.search(r"_(\d{8})$", sess)
    rows = df.index[df.subject_code == subj]
    if not len(rows) or not (n or nf):
        if n or nf: print("no row for", subj, sess)
        continue
    i = (dates[rows] - pd.to_datetime(m.group(1))).abs().idxmin() if m else rows[0]
    df.loc[i, ["n_seg_files", "n_final_files"]] += [n, nf]
    df.loc[i, "seg_session"] = sess
df["segmented"] = df["n_seg_files"].gt(0).map({True: "Yes", False: "No"})
print("segmented rows:", (df.segmented == "Yes").sum(), "| rows with FINAL only:",
      ((df.n_seg_files == 0) & (df.n_final_files > 0)).sum())

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(a.out, index=False)
print(df.shape, "->", a.out)

# filtered copy: drop missing source data (note == "not found"), missing imaging features, no subject id, excluded enchondroma, manual exclusion
keep = ~(df["note"].eq("not found") | df["final_output"].isna() | df["subject_code"].isna()
         | df["enchondroma_decision"].eq("exclude") | df["If_segmented"].eq("exclude"))
filt = Path(a.out).with_suffix(".filtered.csv")
df[keep].to_csv(filt, index=False)
print(keep.sum(), "rows ->", filt)
