
## Data cleaning
Script: [build_combined_dataset.py](./data_ov/build_combined_dataset.py) (cleans and merges the data).

Exclusions:
- Subjects with no imaging data in the source files.
- Enchondromas with max diameter <= 10 mm.
- Subjects whose images have severe noise (e.g., metal implants).

The merged dataset contains the original Spanish reports, English translations, clinical variables, and imaging features extracted by 2 LLM pipelines.

🔴 **TODO: uncertain diagnosis labels.** 78 studies (63 subjects) have several diagnoses in `palabra_manual` (e.g., "osteocondroma, Osteocondromatosis", "encondroma, infarto óseo") and are labelled "uncertain by reports". Not all are checked. Only the cases that enter the selected subset are corrected. Template: [uncertain_labels.csv](../output/label_correction/uncertain_labels.csv) (fill `palabra_manual_corrected`; `build_combined_dataset.py` applies it).

Manual label fixes already made (4 studies) are kept in [manual_label_fixes.csv](../output/label_correction/manual_label_fixes.csv) and re-applied on every merge, so edits made directly in the merged CSV are not lost.

### Subset selection
Aim: select a subset from the cleaned dataset.

Script: [select_subset_waterfill.py](./data_ov/select_subset_waterfill.py)

Latest subset (Nov 6, 15%): [subset_waterfill_filtered1106-15pct.csv](../output/subset/subset_waterfill_filtered1106-15pct.csv) (87 subjects, no uncertain diagnoses). Earlier: `...1103-15pct.csv` (used for BTI v1, v2).

Selection method (even split across diagnosis subtypes):
1. Subjects with an uncertain or missing diagnosis label are left out. Target size T = `--pct`% of the remaining dataset (e.g., 20 -> 20%).
2. Every subtype gets the same share: T / number of subtypes.
3. A subtype smaller than the share contributes all its subjects. A larger one contributes exactly the share.
4. Within a larger subtype, subjects with a segmentation are picked first, then by subject code.

If a subtype has fewer subjects than its share, the missing slots are left empty (not given to other subtypes), so the subset can be smaller than T.

### Subset to BTI input
Aim: build BTI-RADS inputs (MRI + Medium) for the subset from the merged clean data.

Script: [build_bti_input.py](./BTI/build_bti_input.py). Input subset: the 1106 file above. Output: `output/BTI/input_data/` (one row per study).

Fixed rules, no extra LLM call. Anything not available or not clear is left empty.

| BTI field | Source | Rule |
|---|---|---|
| Age | birth date, study date | age in years at the study |
| Size (mm) | LLM2 Max Diameter (longest axis) | number; missing -> manual check |
| Neoplasm history, Suspected metastasis | LLM1 clinical | Yes=1, No=0 |
| Progression | LLM1 clinical | Yes=1, No=0, Not applicable=NA |
| Flat bone, Hand/foot, Spine/pelvis, Body walls | LLM1 Skeletal Location | keyword match (e.g., rib, ilium, L2, metacarpal); empty if location not specified |
| Subperiosteal location, Do not touch | not available | empty |
| Shape (irregular / lobulated / round-oval) | LLM2 Shape | matching one = 1, others 0; "others" or not specified -> all empty |
| Fibrous matrix | LLM2 Fibrous Component | Yes=1, No=0 |
| Fluid without fluid-fluid levels | LLM2 Fluid/Hemorrhage | Yes=1, No=0 |
| Contrast enhancement absence | LLM1 MRI contrast enhancement | "No enhancement"=1, any enhancement=0, "Absence/unknown"=empty |
| Soft tissue invasion | LLM2 Soft Tissue Invasion | Yes=1, No=0 |

Manual review of LLM2 fields (`subset_review.json`): where the verdict is `incorrect`, the corrected value replaces the LLM value. Studies without a review keep the LLM values.

Size: missing cases are listed in `size_missing.csv`. Fill them by hand (`info_key,size_mm`) and pass the file with `--size-manual`. A non-numeric entry (NaN, unknown) excludes that study.

Excluded on purpose (`EXCLUDE` in the script): 3773002 (BONE_AI_172), because our images match study 5032373 of the same subject, where the info was corrected.

Option `--clarify` (default off): treat "not mentioned in the report" as "absent", on the idea that a radiologist reports a feature that is there. Applies to fibrous matrix, fluid without fluid-fluid levels, soft tissue invasion (empty -> 0) and absence of contrast enhancement (empty -> 1). Caveat: an unmentioned enhancement may also mean no contrast was given.

## BTI
Aim: feed the data into the [BTI-RADS 2.0 calculator](https://bti-rads.cic-it-nancy.fr/bti/manage/new) and save its outputs. Code is in the [BTI folder](./BTI).

Calculator usage:
1. Select input mode **MRI** + **Medium**.
2. Fill in the fields to get a prediction.
3. Not all fields are mandatory: the calculator predicts from partial data. Age and size are required.

Outputs per case: BTI-RADS 2.0 score, probability of malignancy (%), and SHAP decision plot.

### Implementation
Script: [bti_predict.py](./BTI/bti_predict.py)
- Input: a CSV whose columns are the form field names (example: `BTI/pseudo.csv`).
- Output: `OUT/<case_id>/result.json` and `shap.png` per case; `OUT/explanation.json` holds the static BTI-RADS legend, which is the same for all cases.
- Queries run sequentially with a delay, stop after 3 failures in a row, and finished cases are skipped on re-run.
- Versions: `--prev <old version>` copies cases whose request is unchanged and queries only new/changed cases. `changes.csv` in the new folder lists copied / rerun / new / dropped cases and the changed fields.

### Versions
Predictions: `output/BTI/prediction/<version>/` (each version is a full copy). Analysis: `output/BTI/analysis/subset_pct15/<version>/` via `compare_bti.py --version vN`.

| Version | Changes | Cases queried |
|---|---|---|
| v1 | first run, default rules | 78 |
| v2 | review corrections (3945099 fluid; 5032373 fluid + size); 3773002 excluded; later 5428038 (BONE_AI_1004) added from the new subset | 78 cases: 2 corrected + 1 added, rest copied from v1 |
| v3 | `--clarify` (unmentioned fibrous matrix / fluid / soft tissue invasion -> 0, unmentioned enhancement -> absence = 1); new subset without uncertain diagnoses | 75 cases: 74 rerun, 1 copied (5428038); 3 dropped |

**Same subjects across versions?** Not exactly, because the subset was re-selected between v2 and v3 (uncertain diagnoses excluded).
- v3 (75 studies, 73 subjects) is fully contained in v2: every v3 case also exists in v2, nothing is new.
- Not in v3 any more: 3308606 (BONE_AI_284), 5032373 (BONE_AI_172) and 5993437 (BONE_AI_173), which the new subset does not select; and 3773002 (BONE_AI_172), excluded on purpose (images match 5032373), which only v1 has.
- v1 (78 studies, 75 subjects) is the old subset. v2 has 76 subjects: v1 minus 3773002 plus 5428038.
- Compare versions on the shared cases only (v3's 75 studies) if the groups should match exactly. Case lists: `changes.csv` in each version folder.

**SHAP images:** until Nov 9 about half of all `shap.png` files (v1-v3) were cut off at the bottom. Cause: `bti_predict.py` replaced every "NaN" in the response with "null", which also changed the base64 image text where it contains "NaN". Fixed in `bti_predict.py` (JSON `parse_constant`); the 108 broken files were repaired with [fix_shap.py](./BTI/fix_shap.py) (checked with PNG checksums). Scores and probabilities were never affected.

### Analysis
Script: [compare_bti.py](./BTI/compare_bti.py) `--version vN [--restrict-to v3]`. Output: `output/BTI/analysis/subset_pct15/<version>/`.
- Diagnosis groups come from `lesion_category.py` (benign, mimicker, intermediate, malignant).
- **Main task:** malignant + intermediate (similar management) vs benign + mimicker. Also reported: malignant vs all others, and malignant vs benign only.
- Metrics (`metrics.csv`): AUC of probability and score (bootstrap 95% CI), and at score >= 4, score >= 5, probability >= 50% and the best Youden cut-off (chosen on the same data, so optimistic): sensitivity, specificity, precision, NPV, F1, accuracy. Plots: ROC per task, confusion matrix (group x score), probability by group and by diagnosis.
- Use `--restrict-to v3` to compare versions on the same studies.

### TODO
- [x] Make the API connection work and save results (Nov 5).
- [x] Create pseudo data (4 cases, all fields filled) to test the workflow (Nov 5). Keep tests few: the server may flag test traffic and block us.
- [x] Run on the real subset (78 studies, no failures) (consider contacting the maintainers before a large batch). [Save](../output/BTI/prediction/)
- [x] Compare and analyse the BTI prediction with the radiological diagnosis (based on report). [Analysis](../output/BTI/analysis/) via [compare_bti.py](./BTI/compare_bti.py) (groups from `lesion_category.py`)
