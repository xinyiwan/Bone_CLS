import os
import glob
from pathlib import Path
from datetime import datetime

base_path = "/home/ext_xinwan/Bone_AI/tmp_data_nifti/ADQUISICIONES"
pattern = os.path.join(base_path, "BONE_AI_*", "*", "segmentation_history")

seg_history_dirs = sorted(glob.glob(pattern))

if not seg_history_dirs:
    print(f"No segmentation_history folders found under {base_path}")
    exit()

rows = []
for seg_dir in seg_history_dirs:
    p = Path(seg_dir)
    subject = p.parent.name
    bone_ai_folder = p.parent.parent.name

    rows.append((bone_ai_folder, subject))

# Print table
col_widths = [
    max(len("Scanner"), max(len(r[0]) for r in rows)),
    max(len("Subject"), max(len(r[1]) for r in rows)),
]


sep = "  ".join("-" * w for w in col_widths)

print(f"\nSegmentation history overview  ({len(rows)} subjects)\n")
print(sep)
for r in rows:
    print(
        f"{r[0]:<{col_widths[0]}}  "
        f"{r[1]:<{col_widths[1]}}  "
    )

print(f"\nTotal subjects with segmentation history: {len(rows)}")
