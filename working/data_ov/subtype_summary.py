"""Per-subtype summary table (n/%, age, top location) in the same shape as the
reference table: one row per subtype grouped under its category (benign /
intermediate / malignant / mimicker / ...), matching lesion_category.py.

Caveats vs. the reference table:
  - No sex column exists in the clinical CSV -- that column is omitted.
  - "Location" here is the first token of localizaciones_encontradas (the
    tokens are Spanish synonyms of the same region, e.g. "Femur, Femoral" --
    taking the first one is a simple proxy, not a full synonym merge).
  - Age = (first scan date - birth date) for each subject, in years.
"""

import argparse
from pathlib import Path

import pandas as pd

from labels import to_label
from lesion_category import to_category

DEFAULT_CSV = Path("/Users/xinyi/Documents/github/Bone_CLS/kira-0515-seg.csv")
OUT_CSV = Path("/Users/xinyi/Documents/github/Bone_CLS/output/clinical_info/subtype_summary_combined.csv")
CATEGORY_ORDER = ("benign", "intermediate", "malignant", "mimicker", "uncertain", "none")
OTHERS_LABEL = "Others"

# --collapse-others keeps only these subtypes per category; the rest fold into "Others".
# Categories not listed here are left fully broken out.
KEEP_SUBTYPES = {
    "benign": {"enchondroma", "osteochondroma", "osteoid osteoma", "bone cyst"},
    "malignant": {"osteosarcoma", "chondrosarcoma", "Ewing sarcoma", "chordoma"},
}


def per_subject_table(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["subject_code"] = df["subject_code"].astype(str).str.strip()
    df = df[df["subject_code"].ne("") & df["subject_code"].ne("nan")]
    df["_label"] = df["palabra_manual"].map(to_label)
    df["_birth"] = pd.to_datetime(df["fechaNaci"], format="%d/%m/%Y", errors="coerce")
    df["_scan"] = pd.to_datetime(df["fechaHoraRealizacion"], errors="coerce", utc=True).dt.tz_localize(None)
    df["_location"] = df["localizaciones_encontradas"].astype(str).str.split(",").str[0].str.strip()

    def collapse(g: pd.DataFrame) -> pd.Series:
        labels = set(g["_label"])
        label = next(iter(labels)) if len(labels) == 1 else "uncertain by reports"
        first = g.sort_values("_scan").iloc[0]
        age = (first["_scan"] - first["_birth"]).days / 365.25 if pd.notna(first["_scan"]) and pd.notna(first["_birth"]) else float("nan")
        return pd.Series({"label": label, "age": age, "location": first["_location"]})

    return df.groupby("subject_code").apply(collapse, include_groups=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("csv", nargs="?", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    ap.add_argument("--collapse-others", action="store_true",
                    help="fold minor subtypes into 'Others' for categories in KEEP_SUBTYPES")
    args = ap.parse_args()

    subjects = per_subject_table(args.csv)
    subjects["category"] = subjects["label"].map(to_category)
    if args.collapse_others:
        for category, keep in KEEP_SUBTYPES.items():
            mask = (subjects["category"] == category) & ~subjects["label"].isin(keep)
            subjects.loc[mask, "label"] = OTHERS_LABEL

    rows = []
    for category in CATEGORY_ORDER:
        cat_df = subjects[subjects["category"] == category]
        if cat_df.empty:
            continue
        cat_total = len(cat_df)
        for label, sub in cat_df.groupby("label"):
            n = len(sub)
            top_loc = sub["location"].value_counts()
            top_loc_str = (f"{top_loc.index[0]} ({top_loc.iloc[0] / n * 100:.1f}%)"
                          if not top_loc.empty and top_loc.index[0] not in ("", "nan") else "")
            rows.append({
                "category": category,
                "subtype": label,
                "n": n,
                "pct_of_category": round(n / cat_total * 100, 2),
                "age_mean": round(sub["age"].mean(), 2) if sub["age"].notna().any() else None,
                "age_sd": round(sub["age"].std(), 2) if sub["age"].notna().sum() > 1 else None,
                "top_location": top_loc_str,
            })
        rows.append({"category": category, "subtype": "Total", "n": cat_total,
                    "pct_of_category": 100.0,
                    "age_mean": round(cat_df["age"].mean(), 2) if cat_df["age"].notna().any() else None,
                    "age_sd": round(cat_df["age"].std(), 2) if cat_df["age"].notna().sum() > 1 else None,
                    "top_location": ""})

    table = pd.DataFrame(rows)
    # Order within a category: named subtypes by n desc, then Others, then Total.
    table["_rank"] = table["subtype"].map({"Total": 2, OTHERS_LABEL: 1}).fillna(0)
    table = table.sort_values(
        ["category", "_rank", "n"], ascending=[True, True, False], kind="stable"
    ).drop(columns="_rank")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.to_string(index=False))
    print(f"\nSaved -> {args.out}")
    print("\nNote: no sex column in the clinical CSV -- sex distribution is omitted.")


if __name__ == "__main__":
    main()
