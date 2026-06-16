from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CSV_PATH = Path("/Users/xinyi/Documents/github/Bone_CLS/kira (Selección 30-03).csv")
OUT_DIR = Path(__file__).resolve().parent
COLUMN = "palabra_manual"

df = pd.read_csv(CSV_PATH)

counts = df[COLUMN].fillna("(missing)").astype(str).str.strip().value_counts()
total = int(counts.sum())

print(f"Loaded {len(df)} rows from {CSV_PATH.name}")
print(f"Unique '{COLUMN}' values: {counts.shape[0]}")
print(f"Non-null entries: {total}\n")

dist = pd.DataFrame({
    "subtype": counts.index,
    "count": counts.values,
    "percent": (counts.values / total * 100).round(2),
})

csv_out = OUT_DIR / "distribution.csv"
dist.to_csv(csv_out, index=False)
print(f"Saved table -> {csv_out}")

print("\nTop 30:")
print(dist.head(30).to_string(index=False))

fig, ax = plt.subplots(figsize=(12, max(6, 0.3 * len(counts))))
ax.barh(dist["subtype"][::-1], dist["count"][::-1], color="steelblue")
ax.set_xlabel("Count")
ax.set_ylabel(COLUMN)
ax.set_title(f"Distribution of '{COLUMN}' (n={total})")
for i, (c, p) in enumerate(zip(dist["count"][::-1], dist["percent"][::-1])):
    ax.text(c, i, f" {c} ({p}%)", va="center", fontsize=8)
plt.tight_layout()

png_out = OUT_DIR / "distribution.png"
plt.savefig(png_out, dpi=150)
print(f"Saved plot  -> {png_out}")
