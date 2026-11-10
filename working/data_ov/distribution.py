import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from labels import to_label

CSV_PATH = Path("/Users/xinyi/Documents/github/Bone_CLS/output/merged_data/kira-1103-combined_dataset.csv")
OUT_DIR = Path("/Users/xinyi/Documents/github/Bone_CLS/output/merged_data")
COLUMN = "palabra_manual"

ap = argparse.ArgumentParser()
ap.add_argument("--csv", type=Path, default=CSV_PATH)
ap.add_argument("--exclude-enchondroma", action="store_true",
                help="drop rows whose manual enchondroma check says 'exclude' (e.g. too small)")
ap.add_argument("--exclude-missing", action="store_true",
                help="drop rows with no source data (note == 'not found') or no imaging features (final_output)")
ap.add_argument("--exclude-no-subject", action="store_true", help="drop rows without subject_code")
ap.add_argument("--exclude-manual", action="store_true",
                help="drop rows manually labelled If_segmented == 'exclude'")
ap.add_argument("--save-filtered", type=Path, help="also write the filtered rows to this CSV")
args = ap.parse_args()
CSV_PATH = args.csv

df = pd.read_csv(CSV_PATH)
n0 = len(df)
drop = pd.Series(False, index=df.index)
suffix = ""
if args.exclude_enchondroma:
    drop |= df["enchondroma_decision"].eq("exclude"); suffix += "_noench"
if args.exclude_missing:
    drop |= df["note"].eq("not found") | df["final_output"].isna(); suffix += "_nomissing"
if args.exclude_no_subject:
    drop |= df["subject_code"].isna(); suffix += "_nosubj"
if args.exclude_manual:
    drop |= df["If_segmented"].str.strip().eq("exclude"); suffix += "_nomanual"
df = df[~drop]
print(f"Filtered {n0} -> {len(df)} rows")
if args.save_filtered:
    df.to_csv(args.save_filtered, index=False); print(f"Saved filtered rows -> {args.save_filtered}")

labels = df[COLUMN].fillna("(missing)").map(to_label)
counts = labels.value_counts()
total = int(counts.sum())

print(f"Loaded {len(df)} rows from {CSV_PATH.name}")
print(f"Unique labels: {counts.shape[0]}")
print(f"Total entries: {total}\n")

dist = pd.DataFrame({
    "subtype": counts.index,
    "count": counts.values,
    "percent": (counts.values / total * 100).round(2),
})

csv_out = OUT_DIR / f"distribution{suffix}.csv"
dist.to_csv(csv_out, index=False)
print(f"Saved table -> {csv_out}")

print("\nDistribution:")
print(dist.to_string(index=False))

fig, ax = plt.subplots(figsize=(12, max(6, 0.3 * len(counts))))
ax.barh(dist["subtype"][::-1], dist["count"][::-1], color="steelblue")
ax.set_xlabel("Count")
ax.set_ylabel("Diagnosis (English)")
ax.set_title(f"Distribution of bone lesion diagnoses (n={total})")
for i, (c, p) in enumerate(zip(dist["count"][::-1], dist["percent"][::-1])):
    ax.text(c, i, f" {c} ({p}%)", va="center", fontsize=8)
plt.tight_layout()

png_out = OUT_DIR / f"distribution{suffix}.png"
plt.savefig(png_out, dpi=150)
print(f"Saved plot  -> {png_out}")
